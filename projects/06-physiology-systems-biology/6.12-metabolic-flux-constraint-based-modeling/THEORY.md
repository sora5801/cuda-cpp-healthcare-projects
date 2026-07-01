# THEORY — 6.12 Metabolic Flux / Constraint-Based Modeling

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

A living cell runs a vast chemical factory: hundreds to thousands of enzyme-
catalysed **reactions** convert nutrients into energy, building blocks, and
biomass. **Metabolic flux** is the *rate* at which each reaction runs. If we knew
every flux, we would know exactly how the cell is using its resources — how fast
it grows, what it excretes, which enzymes are load-bearing.

Measuring every flux directly is impossible, but we can **predict** them from
structure plus a few assumptions. That is **constraint-based modeling**, and its
workhorse is **Flux Balance Analysis (FBA)**. FBA rests on three ideas:

1. **Stoichiometry is known.** From genome annotation we know which metabolites
   each reaction consumes and produces. That is the matrix `S`.
2. **Steady state.** On the timescale of growth, internal metabolites neither
   accumulate nor deplete — what is produced is consumed. Mathematically
   `S v = 0`.
3. **Bounds.** Thermodynamics (some reactions are irreversible), enzyme capacity,
   and nutrient availability cap each flux: `lb ≤ v ≤ ub`.

Among all flux distributions satisfying those constraints, FBA assumes the cell
does something *optimal* — classically, it **maximises growth** (flux through a
"biomass" pseudo-reaction that drains precursors in the ratios needed to build a
new cell). That single optimisation predicts growth rate and a full flux map.

**Why anyone cares.** FBA is the standard tool of systems biology and metabolic
engineering. Two flagship applications, both of which are *many independent FBA
solves*:

- **Gene essentiality screens.** Delete each gene/reaction in turn and ask "can
  the mutant still grow?" Essential reactions are candidate **antibiotic or
  anticancer drug targets** — knock them out and the cell dies. This project
  computes exactly this screen.
- **Condition / drug screens.** Re-solve under thousands of nutrient conditions
  or drug perturbations to predict growth phenotypes.

Both are **embarrassingly parallel over LP instances** — the reason the GPU helps.

## 2. The math

Let there be `m` metabolites and `n` reactions.

- `v ∈ ℝⁿ` — the **flux vector** (units: mmol · gDW⁻¹ · h⁻¹; here arbitrary).
- `S ∈ ℝ^{m×n}` — the **stoichiometry matrix**. `S[i,j]` is how many molecules of
  metabolite `i` reaction `j` produces (`+`) or consumes (`−`); `0` if uninvolved.
- `lb, ub ∈ ℝⁿ` — lower/upper **flux bounds** (a reversible reaction has
  `lb < 0 < ub`; an irreversible one has `lb = 0`).
- `c ∈ ℝⁿ` — the **objective**: `c_j = 1` for the biomass reaction, `0` elsewhere.

FBA is the **linear program (LP)**:

```
maximise    cᵀ v            (growth rate)
subject to  S v = 0         (mass balance at steady state; m equality rows)
            lb ≤ v ≤ ub     (thermodynamic / capacity bounds)
```

A **gene knockout** of reaction `k` is modelled by adding the constraint
`v_k = 0`, i.e. setting `lb_k = ub_k = 0`, and re-solving. The **essentiality
screen** solves this LP `n + 1` times: once per single-reaction deletion, plus the
wild type.

This is a *linear* program because the objective and all constraints are linear in
`v`. Its feasible region is a convex polytope; the optimum (if finite) is attained
at a **vertex**.

## 3. The algorithm — bounded-variable simplex

We solve each LP with the **simplex method**, the classic vertex-walking LP
algorithm (Dantzig, 1947). It starts at a feasible vertex and repeatedly steps to
an adjacent vertex that improves the objective, stopping when none does — that
vertex is optimal.

Textbook simplex assumes variables `x ≥ 0`. FBA variables have **two-sided
bounds** `lb ≤ v ≤ ub`, so we use the **bounded-variable simplex**, which keeps
each nonbasic variable resting at *either* its lower or upper bound.

**Getting a starting vertex.** The equalities `S v = 0` have no obvious feasible
point. We append one **slack** variable per metabolite whose bounds are fixed to
`[0,0]`, turning the system into `[S | I]·[v; s] = 0`. The slack columns form an
identity matrix, so the slacks make an immediate **basic feasible solution** (they
sit at 0, the structural fluxes sit at their lower bounds). That is our start.

**One iteration** (all in `solve_fba()` in `src/fba.h`):

1. **Pricing.** Compute the reduced cost `d_j = c_j − c_Bᵀ (B⁻¹A)_j` of each
   nonbasic variable. For a maximisation, a variable at its *lower* bound improves
   if `d_j > 0`; at its *upper* bound if `d_j < 0`. Pick one to **enter** the
   basis.
2. **Ratio test.** Increase (or decrease) the entering variable. Basic variables
   change linearly; the step stops when the first basic variable hits one of *its*
   bounds (a pivot) or the entering variable reaches its *other* bound (a "bound
   flip", no basis change). Take the smallest such step.
3. **Pivot.** Swap the entering variable in and the blocking variable out;
   Gauss–Jordan-eliminate the entering column so the tableau stays `B⁻¹A`.

Repeat until pricing finds no improving variable → **optimal**.

**Pivot rule = Bland's rule.** Among improving variables we always take the
**lowest index**. Bland's rule provably prevents *cycling* (the simplex getting
stuck looping among degenerate vertices) and — crucially for us — makes every
choice **deterministic**. Same rule on CPU and GPU ⇒ same pivots ⇒ same answer.

**Complexity.** Each iteration is `O(m · (n+m))` work (price all columns, pivot
one row on a dense `m × (n+m)` tableau). The number of iterations is small for
these tiny models (the demo converges in **5** iterations). Worst-case simplex is
exponential in pathological problems, but on real metabolic LPs it is fast and
robust; production solvers add anti-degeneracy and scaling refinements (§7).

```
Per LP:  build [S|I]  ->  [ price -> ratio-test -> pivot ]*  ->  read cᵀv
                              \___ a few iterations ___/
Screen:  (n+1) such LPs, all INDEPENDENT  ->  one GPU thread each
```

## 4. The GPU mapping

The parallelism is at the **LP level**, not inside one LP. The knockout screen is
`n + 1` completely independent LP solves (deleting reaction 3 has nothing to do
with deleting reaction 7), so:

- **Thread-to-data mapping.** Thread `k` solves the LP with reaction `k` deleted;
  thread `n` (the last job) solves the wild type. `k = blockIdx.x·blockDim.x +
  threadIdx.x`, guarded against the ragged last block. See `screen_kernel` in
  `src/kernels.cu`.
- **Launch configuration.** `block = 64` threads, `grid = ⌈(n+1)/64⌉`. Why only
  64 and not the usual 256? **Per-thread state.** Each thread runs a whole simplex
  with its own dense tableau and bound/basis arrays in **local memory** (~3–5 KB;
  see `FBA_MAX_*` in `fba.h`). Packing 256 such threads per block would spill
  registers and thrash local memory. 64 keeps enough warps resident to hide
  latency while leaving headroom — an honest **occupancy-vs-footprint** lesson.
- **Memory hierarchy.** This teaching version keeps everything **per-thread
  local**: no shared memory, no atomics, no synchronisation. That maximises
  independence and clarity at the cost of a large local footprint (and thus capped
  occupancy). The `model` struct is passed **by value** into the kernel, so each
  thread gets its own copy to clamp — no device allocation of inputs needed.

```
grid ──> block(64 threads) ──> thread k
                                  │  private FbaModel copy (clamp reaction k)
                                  │  private simplex tableau in local memory
                                  └─ solve_fba()  ->  out[k].objective
   (nrxn+1) threads total, zero inter-thread communication
```

**No CUDA library is used here** — the solver is hand-written on purpose, so the
LP method is not a black box. The catalog's suggested `cuSOLVER dense LP factor` /
`custom interior-point` and the "one LP per **block** with shared-memory tableau"
layout are the *production* design (§7): there, the constraint matrix is staged in
**shared memory** and a whole block cooperates on one large LP, with **warp-level
reductions** for the pricing step. Our "one LP per **thread**" layout is the right
call for *many small* LPs; the block-per-LP layout is right for *fewer, larger*
LPs. THEORY §7 and the exercises in the README explore the crossover.

## 5. Numerical considerations

- **Precision: FP64 (double).** Simplex does repeated Gauss–Jordan elimination;
  accumulated round-off in FP32 can corrupt pivot decisions and feasibility. We
  use `double` throughout `fba.h`. On consumer GPUs FP64 is slower than FP32, but
  correctness dominates for a solver, and the models are tiny.
- **Determinism.** No atomics, no parallel reductions across threads — each thread
  owns its LP end to end — so there is **no floating-point reordering** between
  runs. Combined with Bland's rule, the output is bit-stable run to run.
- **CPU/GPU parity.** Because the CPU reference and the GPU kernel call the
  *identical* `solve_fba()` from `fba.h`, they execute the same operations in the
  same order. The one theoretical wrinkle is **FMA contraction**: the GPU may fuse
  a multiply-add that the host compiler rounds in two steps, perturbing a reduced
  cost by ~1e-16. On this well-conditioned integer-stoichiometry model that never
  flips a pivot, and the observed CPU–GPU difference is exactly **0**. We still
  verify to a documented `1e-9` tolerance rather than claim bit-identity (see §6).
- **Degeneracy caveat.** LPs can have multiple optimal vertices (alternate optima)
  with the *same* objective but *different* flux vectors. If a reduced cost sits
  right at the pricing threshold, FMA differences *could* in principle steer CPU
  and GPU to different (equally optimal) vertices. We compare the **objective**
  (the biologically meaningful, unique quantity), which is robust to this; a flux-
  by-flux comparison would need flux-variability analysis (an exercise).

## 6. How we verify correctness

Three independent checks:

1. **CPU reference (`src/reference_cpu.cpp`).** `screen_cpu()` solves the same
   `n+1` LPs in a plain serial loop. `main.cu` compares its objective array to the
   GPU array element-by-element; the worst absolute difference must be
   `≤ 1e-9` (reported on stderr as `worst |CPU-GPU| objective diff`, observed
   `0`). Two independent code paths agreeing is strong evidence of correctness.
2. **Tolerance rationale (PATTERNS.md §4).** This is the "same exact operations on
   both sides" case, so the honest expectation is machine-precision agreement; we
   set `1e-9` to absorb any FMA reassociation without masking a real bug.
3. **Known-answer / analytic check.** The toy network is engineered so the answers
   are readable by hand: wild-type growth `= 10` (the uptake bound); three
   reactions with no alternative route are **essential** (growth `0`); the isozyme
   pair and the overflow are **neutral**; and deleting `B->C` forces the capacity-3
   bypass, giving exactly `3` (a **30%** partial knockout). The program's summary
   (`3 essential, 1 reducing, 4 neutral`) matches this by construction — validating
   the *science*, not just CPU==GPU agreement.

## 7. Where this sits in the real world

This is a deliberately **reduced-scope teaching version** (CLAUDE.md §13). The
math (`max cᵀv s.t. Sv=0, lb≤v≤ub`) and the parallelism (many independent LPs) are
*exactly* those of production FBA; what differs is scale and solver sophistication:

- **Model size.** Genome-scale models (**Recon3D** ~10 600 reactions; **BiGG**
  models) have thousands of reactions and are **~99% sparse**. Real solvers store
  `S` as a sparse matrix (CSR) and never form a dense tableau; our dense
  `FBA_MAX_*` layout would be hopeless at that scale.
- **Solver.** Production tools use the **revised simplex** (which factorises the
  basis `B` and updates `B⁻¹` implicitly, avoiding full tableau pivots) or, more
  often, **interior-point / barrier** methods that march through the polytope
  interior and scale far better to large sparse LPs. The catalog notes
  GPU-accelerated interior-point primal–dual kernels for LP batches.
- **Tooling.** **COBRApy** wraps industrial LP/MILP backends (HiGHS, Gurobi,
  CPLEX). On top of plain FBA it adds **FVA** (flux variability — the min/max of
  each flux at optimal growth), **pFBA** (parsimonious FBA — minimise total flux
  among optima), **thermodynamic FBA**, **MILP gap-filling**, and **minimal cut
  sets**. Each is a variation on the same LP core.
- **GPU layout at scale.** For *large* LPs the win is one **LP per block** with the
  constraint matrix in **shared memory** and **warp reductions** for pricing (the
  catalog's suggested pattern), rather than our one-LP-per-thread scheme that suits
  *many small* LPs. Choosing between them is a real engineering decision governed
  by LP size vs. LP count.

None of these change the concepts you learn here — they change the data
structures and the inner solver.

---

## References

- Orth, Thiele & Palsson, *"What is flux balance analysis?"*, **Nature
  Biotechnology** 28:245 (2010) — the canonical one-page introduction; read first.
- Dantzig, *Linear Programming and Extensions* (1963); Bland, *"New finite pivoting
  rules for the simplex method"*, Math. of OR (1977) — the anti-cycling pivot rule
  we use.
- Maros, *Computational Techniques of the Simplex Method* (2003) — the bounded-
  variable simplex in full detail.
- **COBRApy** — <https://github.com/opencobra/cobrapy> — the reference Python
  toolkit; study `model.optimize()`, `single_reaction_deletion`, and
  `flux_variability_analysis` to see the production versions of this project.
- **BiGG Models** <http://bigg.ucsd.edu>, **Recon3D**
  <https://github.com/SBRG/Recon3D>, **Virtual Metabolic Human**
  <https://vmh.life> — real genome-scale models to scale up to.
- **SUNDIALS** <https://github.com/LLNL/sundials> — ODE integrators used for
  *dynamic* FBA (dFBA), where FBA is re-solved along a time course.
