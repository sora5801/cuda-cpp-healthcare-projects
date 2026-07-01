# THEORY — 6.8 Tumor Growth & Treatment-Response Modeling

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to mathematical oncology. See
> [README.md](README.md) for the quick tour and build steps.
>
> _Educational only — not for clinical use. All data here is synthetic._

---

## 1. The science

A solid tumor is a population of cells that (a) **proliferate** (divide) and
(b) **infiltrate** the surrounding tissue by random motility. Left alone, a small
avascular tumor grows outward as a roughly circular mass with a dense core and a
diffuse, infiltrating rim — the rim is exactly why surgery so often "misses"
cells and the tumor recurs. **Mathematical oncology** models this so clinicians
can ask *what-if* questions: how fast will it grow, how far have cells already
spread beyond what imaging shows, and how much will a given **radiotherapy**
schedule set it back?

This project builds the smallest model that captures those dynamics honestly:

- **Growth + spread** with the **Fisher-KPP reaction-diffusion equation** — the
  canonical PDE of glioma modeling (Swanson et al.'s "proliferation-invasion" or
  *PI* model). A single scalar field `u(x,y)` in `[0,1]` is the local tumor cell
  density normalized to the tissue carrying capacity (`u = 1` ⇒ packed with tumor
  cells, `u = 0` ⇒ none).
- **Treatment response** with the **linear-quadratic (LQ) radiobiological model**
  — the single most-used equation in the radiotherapy clinic. Each radiation
  fraction kills a fixed *fraction* of cells everywhere; the survivors regrow.

We deliberately keep it 2-D and single-field (a **reduced-scope teaching
version**, CLAUDE.md §13). Section 7 describes what a production model adds
(oxygen/hypoxia fields, drug PK/PD coupling, 3-D, agent-based cells).

## 2. The math

### 2.1 Growth + spread — Fisher-KPP

The tumor density evolves by

```
∂u/∂t = D ∇²u  +  ρ u (1 − u)
        \_____/   \_________/
        diffusion   logistic growth (reaction)
```

| Symbol | Meaning | Units |
|---|---|---|
| `u(x,y,t)` | local tumor cell density / carrying capacity | dimensionless, `[0,1]` |
| `D` | cell diffusion (infiltration) coefficient | mm²/day |
| `ρ` | net proliferation rate | 1/day |
| `∇²` | Laplacian (spatial diffusion operator) | 1/mm² |

The **reaction** term `ρ u(1−u)` is *logistic*: growth is fast while the tumor is
sparse (`u ≪ 1`) and saturates to zero as `u → 1` (cells run out of room and
nutrients). The **diffusion** term `D∇²u` spreads cells into neighbouring tissue.

A classic result: a Fisher-KPP front (the boundary between tumor and healthy
tissue) settles into a **travelling wave** moving at the constant speed

```
c = 2 √(D ρ)      [mm/day]
```

For the sample (`D = 0.02`, `ρ = 0.15`), `c = 2√(0.003) ≈ 0.110 mm/day`. Over
100 days that predicts the front advancing ~11 mm from the seed — the demo's
untreated core radius (`≈ 9 mm`, plus the diffuse rim) is consistent with this
analytic speed, which is our **science-level** sanity check (docs/PATTERNS.md §4).

### 2.2 Treatment — the linear-quadratic model

A single radiotherapy fraction delivering physical dose `d` [Gy] leaves a
**surviving fraction** of clonogenic cells

```
S(d) = exp( −(α d + β d²) )
```

| Symbol | Meaning | Units |
|---|---|---|
| `α` | linear (single-hit) radiosensitivity | 1/Gy |
| `β` | quadratic (two-hit) radiosensitivity | 1/Gy² |
| `α/β` | the dose at which linear and quadratic kill are equal | Gy |

`α/β ≈ 10 Gy` for most tumors and early-responding tissue, `≈ 3 Gy` for
late-responding normal tissue — the number that justifies **fractionation**
(splitting the dose into many small fractions spares normal tissue more than
tumor). For a schedule of `n` fractions of dose `d`, radiobiologists summarize the
"punch" with the **biologically-effective dose**

```
BED = n d ( 1 + d / (α/β) )     [Gy]
```

Because radiation cell-kill (seconds) is far faster than growth (days), we apply
each fraction as an **instantaneous multiplicative kill**: `u ← S(d)·u` at every
voxel, on the scheduled timestep. Between fractions the tumor regrows by §2.1.

## 3. The algorithm

We integrate the PDE with **explicit (forward) Euler** in time and the standard
**5-point finite-difference Laplacian** in space, on an `nx × ny` grid with
periodic boundaries. One timestep, per cell `(x,y)` with flat index `i = y·nx+x`:

```
lap5(u)_i = u[x−1,y] + u[x+1,y] + u[x,y−1] + u[x,y+1] − 4 u[x,y]
u_new_i   = u_i + Δt ( (D/Δx²) lap5(u)_i + ρ u_i (1 − u_i) )
u_new_i   = clamp(u_new_i, 0, 1)          # keep the density physical
```

The clamp removes tiny non-physical overshoots (`u < 0` or `u > 1`) that explicit
Euler can produce right at the sharp front. On a **fraction step** we first do the
per-cell kill `u_i ← S(d)·u_i` (see `is_fraction_step`), then the growth update.

**Double buffering (ping-pong).** Every cell reads its neighbours' *old* values
and writes a *new* value. If we wrote in place, a cell updated early would corrupt
the input of its neighbour. So we keep two buffers, read all of "current", write
all of "next", then swap — no read/write hazard.

**Complexity.** Each of `S` steps touches all `N = nx·ny` cells with `O(1)` work
(5 loads, a handful of flops), so the whole simulation is `O(S·N)` time. Serially
that is one long loop; in parallel the **work** is still `O(S·N)` but the **depth**
is `O(S)` — every cell within a step is independent, so a step is one parallel
sweep. This is why the GPU helps: `N` is large and every cell of a step can run at
once.

## 4. The GPU mapping

This is the **stencil + ping-pong** pattern (docs/PATTERNS.md §1), the same shape
as the lattice-Boltzmann flagship (6.04) and the reaction-diffusion project
(14.02). The per-cell physics lives once in `src/tumor.h` as `__host__ __device__`
inline functions, so the CPU reference and the GPU kernel run **identical math**
(the HD-macro idiom, PATTERNS.md §2).

- **Thread-to-data mapping.** One thread per grid cell. A 2-D block of
  `16×16 = 256` threads tiles the field; thread `(x,y)` with
  `x = blockIdx.x·blockDim.x + threadIdx.x` (and similarly for `y`) owns cell
  `(x,y)` and computes `u_new[i]` from `u[i]` and its four neighbours.
- **Launch configuration.** `block = (16,16)`; `grid = (⌈nx/16⌉, ⌈ny/16⌉)`. 256
  threads/block is a solid occupancy default on sm_75–sm_89 (8 warps to hide
  memory latency). A square tile keeps each thread's four neighbour loads spatially
  close in the row-major field, which helps coalescing.
- **Treatment kernel.** A separate flat 1-D launch over `N` cells: thread `i` does
  `u[i] *= S`. Purely local, embarrassingly parallel, no buffers to swap.
- **The host time loop** stays on the CPU: for each step it launches the treatment
  kernel (if scheduled), then the growth kernel current→next, then swaps the two
  device pointers. Many small launches make this **launch-bound** on tiny grids
  (see §7 timing note).
- **Memory hierarchy.** This teaching version uses only **global memory** plus
  registers. Each cell is read ~5× per step (once as centre, four times as a
  neighbour of its neighbours). The obvious optimization is **shared-memory
  tiling**: each block stages its `16×16` cells plus a 1-cell **halo** into
  `__shared__` memory, so those redundant global reads become fast shared reads.
  We keep the naive version because it teaches the stencil clearly; the tiled
  variant is left as an exercise.

```
Grid decomposition (nx × ny cells, 16×16 tiles):

        block(0,0)      block(1,0)   ...
      +-------------+-------------+
      | 16x16 cells | 16x16 cells |
      | 1 thread /  |             |
      |   cell      |             |     each thread reads its 4 neighbours
      +-------------+-------------+     (periodic wrap at the edges) from the
      | block(0,1)  |             |     FROZEN "current" buffer and writes
      |             |             |     one cell into the "next" buffer;
      +-------------+-------------+     host swaps the two buffers each step.
```

Nothing here needs a CUDA library: the Laplacian, the logistic reaction, and the
LQ multiply are all hand-written custom kernels — exactly the "custom FD stencil
kernels" the catalog calls for. (A production 3-D code would additionally use
**Thrust** to sort/bin agent-based cells and **cuRAND** for stochastic
division/death; see §7.)

## 5. Numerical considerations

- **Precision: FP64 (double).** Reaction-diffusion accumulates many small
  increments over hundreds of steps; double precision keeps the front position
  and the burden accurate and makes CPU/GPU agreement essentially exact.
- **Stability (CFL / diffusion number).** Explicit Euler for 2-D diffusion is
  only stable if `Δt ≤ Δx² / (4D)`. `load_tumor` enforces this and refuses an
  unstable config (the sample's `Δt = 0.25 ≤ 0.04/0.08 = 0.5 day`). Violating it
  makes the field oscillate and blow up — a classic, worth-seeing failure mode.
- **No atomics, no races.** Within a step every thread writes a *distinct* output
  cell and only *reads* the frozen input buffer, so there is no contention and no
  need for atomics. Determinism is automatic — the order threads run in cannot
  change any result.
- **Determinism.** stdout is byte-identical every run (verified). Timings go to
  stderr. There are no floating-point reductions with data-dependent order, so
  (unlike the Monte-Carlo/k-means projects) we did not need integer/fixed-point
  accumulation.

## 6. How we verify correctness

Two independent checks (docs/PATTERNS.md §4):

1. **CPU == GPU.** `src/reference_cpu.cpp` runs the identical `tumor.h` physics in
   a plain serial loop. `main.cu` runs both the treated and the untreated-control
   scenarios and compares the final density fields cell by cell. The tolerance is
   `1e-6` (documented in `main.cu`). In practice the worst difference on the
   sample is `~2e-16` (machine epsilon) because both paths execute the same
   double-precision operations in the same order; we keep the tolerance at `1e-6`
   so the test stays robust across GPU architectures, where fused-multiply-add
   (FMA) contraction can legitimately differ at the `~1e-6` level over thousands of
   nonlinear steps. Densities are `O(1)`, so `1e-6` is physically negligible — the
   *same* tumor. We do **not** claim bit-identity.
2. **Analytic / science check.** The Fisher wave speed `c = 2√(Dρ) ≈ 0.11 mm/day`
   predicts the untreated front's reach over the run (~11 mm), consistent with the
   reported core radius; and the LQ surviving fraction printed by the program
   (`S = 0.6977`) matches `exp(−(0.15·2 + 0.015·4)) = exp(−0.36)` by hand. These
   validate the *science*, not just CPU/GPU agreement.

**Edge cases:** the loader rejects bad shapes, negative physics, and unstable
timesteps; `n_fractions = 0` is a valid untreated control (which the program runs
internally).

## 7. Where this sits in the real world

This is a deliberately reduced teaching model. Production mathematical-oncology
platforms add:

- **Coupled fields.** Real tumors are limited by **oxygen and nutrients**: an
  extra diffusion-consumption PDE for O₂ drives **hypoxia** (low-oxygen regions
  grow slowly and are *radioresistant* — the LQ `α`,`β` shrink where O₂ is low)
  and **necrosis** in the starved core. Chemotherapy adds a **drug PK/PD** field
  (a concentration PDE coupled to a pharmacodynamic kill term). Each field is
  another stencil on the same grid — the pattern scales directly.
- **Agent-based cells.** **PhysiCell** and **PhysiBoSS** track *individual* cells
  (position, cycle state, an intracellular Boolean signaling network via MaBoSS)
  diffusing in a substrate field — capturing heterogeneity a continuum density
  cannot. **Chaste** includes off-lattice spheroid and crypt models. On the GPU
  these use **Thrust** to sort/bin cells into a spatial grid and **cuRAND** for
  stochastic division/death — the "separate agent kernel with shared-memory
  neighbourhood queries" of the catalog.
- **3-D and scale.** Clinical models run on `256³–512³` voxel grids (a 512³ grid
  is `1.3×10⁸` cells) for many simulated days, and sweep **thousands of parameter
  sets** for *in-silico* clinical trials — embarrassingly parallel across the GPU
  and across GPUs. Calibration uses **TCGA** omics and **TCIA** imaging.
- **Better numerics.** Implicit or IMEX time-stepping (removing the `Δt ≤ Δx²/4D`
  limit), phase-field or level-set tracking of the tumor boundary, and adaptive
  meshes near the front.

**Timing note (honesty rule, PATTERNS.md §7).** The reported GPU/CPU ms are a
*teaching artifact*, not a benchmark. On this tiny `128²` grid the ~400 small
kernel launches make the loop partly launch-bound; the GPU still wins here (~10×),
and its edge grows sharply with grid size — which is the whole reason 3-D clinical
models need the GPU.

---

## References

- **Fisher (1937)** and **Kolmogorov–Petrovsky–Piskunov (1937)** — the original
  reaction-diffusion "traveling wave" papers behind Fisher-KPP.
- **Swanson, Alvord & Murray (2000)**, *Cell Proliferation* — the
  proliferation-invasion (PI) glioma model this project implements in miniature.
- **Fowler (1989)**, *Br. J. Radiol.* — the LQ model and BED; the standard
  reference for fractionated radiotherapy radiobiology.
- **PhysiCell** — <https://github.com/MathCancer/PhysiCell> — 3-D agent-based
  multicellular simulator with diffusing substrates; study its BioFVM diffusion
  solver and how it scales linearly in cell count.
- **PhysiBoSS** — <https://github.com/PhysiBoSS/PhysiBoSS> — PhysiCell + MaBoSS
  Boolean intracellular signaling; the model of intracellular decision logic.
- **Chaste** — <https://github.com/Chaste/Chaste> — tumor spheroid and crypt
  models; a mature C++ computational-biology framework.
- **OpenFOAM** — <https://github.com/OpenFOAM/OpenFOAM-dev> — CFD used for
  drug-delivery flow simulations coupled to tumor models.
