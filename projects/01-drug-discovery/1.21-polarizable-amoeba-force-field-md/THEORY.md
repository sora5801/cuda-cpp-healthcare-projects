# THEORY — 1.21 Polarizable / AMOEBA Force Field MD

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

A **force field** is the function that gives a molecular-dynamics (MD) simulation
its energy and forces. The cheap, ubiquitous force fields (AMBER, CHARMM, OPLS)
are **fixed-charge**: every atom carries a constant partial charge, baked in when
the parameters were fit. That is a real physical approximation — it pretends an
atom's electron cloud never deforms in response to its surroundings. But electron
clouds *do* deform: an atom in a strong local field develops an **induced dipole**.
This is **polarization**, and it matters most exactly where drug discovery cares:
charged ligands, metal ions, buried polar pockets, and the **binding free
energies** and **pKa shifts** that decide whether a molecule is a drug.

The **AMOEBA** force field ("Atomic Multipole Optimized Energetics for
Biomolecular Applications") restores polarization. Each atom carries permanent
**multipoles** (charge + dipole + quadrupole) *and* an **induced dipole** `μ_i`
that responds to the total electric field at that atom. The catch: the field at
atom `i` includes the dipoles of all the other atoms, and *their* dipoles depend
on the field at *them*, which includes `μ_i`. The dipoles are mutually coupled, so
they must be found by solving a **self-consistent field (SCF)** problem — and this
must be redone at **every MD timestep**, because the atoms moved. That SCF solve is
what makes AMOEBA ~10× costlier than AMBER, and it is the single computation this
project implements and parallelizes.

**The question we answer numerically:** given a frozen snapshot of atoms (their
positions, polarizabilities, and the permanent field acting on them), what are the
self-consistent induced dipoles `μ_i`, and what polarization energy do they store?

## 2. The math

**Induced-dipole relation.** An atom of isotropic polarizability `α_i`
(units: Å³) placed in a total field `E_i^tot` acquires a dipole

```
μ_i = α_i · E_i^tot
```

The total field is the **permanent** field `E_i^perm` (from fixed charges /
external sources — our right-hand side) plus the field produced by every *other*
induced dipole:

```
μ_i = α_i ( E_i^perm + Σ_{j≠i} T_ij · μ_j )                       (1)
```

**Dipole–dipole interaction tensor.** With `r = r_i − r_j` (vector from `j` to
`i`) and `r = |r|`, the 3×3 tensor that turns a dipole on `j` into a field at `i`
is

```
T_ij = ( 3 r rᵀ − r² I ) / r⁵                                    (2)
```

so `T_ij · μ_j = (3 (r̂·μ_j) r̂ − μ_j) / r³`. (Implemented as
`dipole_field_contrib` in `src/amoeba.h`.)

**Linear system.** Divide (1) by `α_i` and move the coupling to the left:

```
(1/α_i) μ_i − Σ_{j≠i} T_ij · μ_j = E_i^perm                       (3)
```

Stack the `N` atoms' 3-vectors into one `3N`-vector. Then (3) is a **linear
system**

```
A μ = b ,   b_i = E_i^perm ,
A = blockdiag(1/α_i) − T   (T the off-diagonal dipole-coupling)  (4)
```

`A` is **symmetric** (because `T_ij = T_jiᵀ` and each `T_ij` is itself symmetric)
and, for physical polarizabilities and non-overlapping atoms, **positive
definite** (the diagonal `1/α_i` dominates the weak `1/r³` coupling). Symmetric +
positive-definite (SPD) is precisely the class the **conjugate gradient** method
was designed for.

**Polarization energy.** Once `μ` is known,

```
U_pol = −½ Σ_i μ_i · E_i^perm                                     (5)
```

The factor ½ is the "self-energy" cost of polarizing the atoms (half the
interaction energy is spent doing the work of inducing the dipole). This scalar is
what we report and verify.

**Symbols:** `μ_i` induced dipole [dipole units]; `α_i` polarizability [Å³];
`E_i^perm` permanent field [field units]; `r_i` position [Å]; `T_ij` interaction
tensor [1/Å³]; `U_pol` polarization energy [energy units]. (Reduced/illustrative
units — this is a teaching model.)

## 3. The algorithm

We solve `A μ = b` with **matrix-free conjugate gradient (CG)**. CG is the
canonical Krylov solver for an SPD system: it builds search directions that are
**A-orthogonal** (conjugate), so it reduces the error optimally using only
**matrix-vector products** `A·p` — it never needs the matrix `A` itself.

```
μ ← α_i E_i^perm                  # warm start: the uncoupled (fixed-charge) guess
r ← b − A μ                       # residual
p ← r
rs_old ← ⟨r, r⟩
repeat until ⟨r,r⟩ ≤ tol²·⟨b,b⟩ or max_iter:
    Ap ← A p                      # the ONE matvec per iteration  (apply_A)
    α  ← rs_old / ⟨p, Ap⟩         # exact line-search step length
    μ  ← μ + α p
    r  ← r − α Ap
    rs_new ← ⟨r, r⟩
    β  ← rs_new / rs_old          # Fletcher–Reeves: keeps directions conjugate
    p  ← r + β p
    rs_old ← rs_new
```

This is exactly `solve_induced_dipoles` in `src/amoeba.h`.

**Why CG and not fixed-point iteration?** The textbook "Jacobi" approach just
re-evaluates (1) repeatedly (`μ ← α(E + Tμ)`). That converges only when the
coupling is weak and does so **linearly**. CG converges in at most `3N` exact
steps and, for the tightly-clustered spectrum of a weakly-coupled dipole system,
in a *handful* — which is why our 3-atom demo converges in 2 iterations (honest;
the spectrum of `A` is nearly `1/α·I`). Warm-starting at the uncoupled dipoles is
also what production codes do between MD steps.

**Complexity.** One CG iteration does one matvec (`apply_A`) plus two dot products
and three vector updates. The matvec is the all-pairs loop in (2): **O(N²)** per
iteration for an `N`-atom system. With `k` iterations, one system costs
**O(k·N²)**. The *serial* cost of the whole ensemble of `M` systems is
**O(M·k·N²)**. The parallel work is the same; the **depth** (critical path) for
one thread is `O(k·N²)` and the `M` systems run concurrently — so wall-clock time
falls as `~1/min(M, #cores)`.

**Arithmetic intensity / access pattern.** The matvec re-reads the system's
positions and the current vector `N` times each — but for our small systems that
all fits in registers/local memory, so there is essentially no global-memory
traffic inside the CG loop. The pattern is compute-bound per thread, latency-bound
across the ensemble (each thread is an independent sequential solve).

## 4. The GPU mapping

**Pattern:** *ensemble of independent solves — one thread per system*
(PATTERNS.md "ensemble" row; flagship 9.02 is the exemplar). Each MD step / sweep
member is an independent `A μ = b`, so we hand each its own thread; the thread runs
the **entire** CG loop in its registers/local memory and writes one
`PerSystemResult`. No shared memory, no atomics, no cross-thread dependence.

**Thread-to-data mapping.** `idx = blockIdx.x · blockDim.x + threadIdx.x` owns
ensemble member `idx`. The ragged last block is guarded with `if (idx >= M)
return;`.

**Launch configuration.** `block = 128` threads (a warp multiple; modest so the
register-heavy per-thread CG state does not crush occupancy on sm_75..sm_89);
`grid = ceil(M / 128)`. See `THREADS_PER_BLOCK` in `src/kernels.cu`.

**Memory hierarchy and why.**
- **Registers / local memory** hold the whole CG working set (`μ, r, p, Ap`, each
  `AMOEBA_MAX_ATOMS × 3` doubles). This is where all the work happens — fast, and
  private to the thread. We cap atoms at `AMOEBA_MAX_ATOMS = 32` so this footprint
  stays small (large per-thread arrays spill to slow local memory).
- **Global memory** holds only the input `AtomSystem[]` (read once into a local
  copy) and the output `PerSystemResult[]` (written once). One contiguous H2D copy
  in, one D2H copy out.
- **No shared / constant / texture memory** is needed for thread-per-system: there
  is nothing to share between threads. (Constant memory would help if every thread
  read the *same* query — that is the `1.12` pattern, not this one.)

```
ensemble of M independent A μ = b solves
        │
   ┌────┴───────────────── grid (ceil(M/128) blocks) ──────────────────┐
   block 0                 block 1                       block g-1
 ┌──────────┐            ┌──────────┐                  ┌──────────┐
 │ t0 t1 .. │            │ ..       │        ...       │       .. │
 └─┬─┬──────┘            └──────────┘                  └──────────┘
   │ └ thread t1: member 1 → full CG loop → result[1]
   └── thread t0: member 0 → full CG loop → result[0]
        (each thread: warm start → {matvec, 2 dots, 3 axpy} × k → energy)
```

**Which CUDA library does what — and what hand-rolling takes.** *None.* The
catalog names "custom CUDA conjugate-gradient solver," and we write it by hand so
nothing is a black box — the CG loop, the SPD matvec, and the dot products are all
explicit in `amoeba.h`. The catalog also lists **cuFFT** (for multipole-PME Ewald
of the *permanent* electrostatics) and **NCCL** (multi-GPU) — those belong to the
full production stack (§7), not to the induced-dipole SCF we isolate here. If we
*did* offload a step: a cuFFT call computes the discrete Fourier transform of the
charge/multipole grid (the reciprocal-space part of Ewald); hand-rolling it means
writing a radix-2/3/5 mixed FFT with the right twiddle factors and bit-reversal —
a project in itself (see flagship `8.03` for a from-scratch-vs-cuFFT treatment).

**A larger-system alternative (exercise 3).** When `N` is large, thread-per-system
underuses the GPU. The right mapping is **block-per-system**: a thread block
cooperates on one CG solve — split the `O(N²)` matvec across the block's threads
and use a **block reduction** (shared memory + `__syncthreads`, or
`cub::BlockReduce`) for the two dot products. That is how Tinker-HP/OpenMM
parallelize a *single* big system.

## 5. Numerical considerations

- **Precision: FP64 throughout.** Polarization solves are sensitive — the `1/r⁵`
  tensor and the difference `(3 r rᵀ − r² I)` lose significance in single
  precision, and CG's conjugacy degrades when dot products are noisy. We use
  `double` everywhere in `amoeba.h`. Exercise 5 invites you to try FP32 and watch
  the needed tolerance balloon.
- **Determinism.** Every reduction in this project is a **fixed-order** serial sum
  (`dot3N` walks atoms 0..n−1, components 0..2 in the same order on host and
  device), and each thread owns its system alone — there are **no atomics** and no
  cross-thread floating-point reductions whose order could vary. So stdout is
  **byte-identical every run** (PATTERNS.md §3), which is what lets the demo diff
  it. (Contrast: a block-per-system matvec reduction *would* reorder the sum and
  need fixed-point or a tree-reduction to stay deterministic.)
- **The GPU's FMA.** The device contracts `a*b+c` into a single fused
  multiply-add, which the host may not. Over a couple dozen CG iterations this
  produces a divergence of ~`1e-16`–`1e-17` between CPU and GPU — far below any
  physical scale (§6).
- **Stability / the polarization catastrophe.** If polarizabilities are too large
  or atoms too close, `A` loses positive-definiteness, the mutual field runs away,
  and `μ` blows up — the classic failure mode real AMOEBA tames with **Thole
  damping** of short-range dipole interactions. We report `max|μ|` as a diagnostic
  and leave damping as an exercise. `max_iter` caps the loop so a non-convergent
  case fails loudly rather than spinning.

## 6. How we verify correctness

The CPU reference (`src/reference_cpu.cpp` → `integrate_cpu`) solves every system
with the **same** `solve_induced_dipoles` routine the GPU kernel calls — the code
is shared host+device through the `AMOEBA_HD` macro in `amoeba.h`. `main.cu` then
compares, per member, the polarization energy `U_pol` and the net induced dipole,
and reports the worst absolute difference.

**Tolerance: `1.0e-9`.** Because both sides run the identical double-precision
operations, they agree to machine precision; the only divergence is the GPU's FMA
contraction (§5), worth ~`1e-17` here. `1e-9` is therefore a comfortable,
**honest** bound — orders of magnitude tighter than any physically meaningful
scale, yet loose enough to never flake on FMA differences (PATTERNS.md §4, the
"exact-same-ops" case). The demo prints the *actual* worst diff (~`5.6e-17`) on
stderr so you can see how much head-room there is.

Why is "an independent serial implementation agrees with the GPU" convincing? It
isn't a self-fulfilling check here because the *interesting* failure modes — a
race, an out-of-bounds index, a launch-config bug, a wrong stride into the global
arrays — would corrupt the GPU result while leaving the CPU result intact, so they
*would* break the comparison. The shared math removes only the *uninteresting*
disagreement (different rounding), which is exactly what we want to ignore.

**A second, physical check (exercise 4):** for a single isolated atom the coupling
vanishes and the exact answer is `μ = α E`; asserting against that closed form
validates the *science*, not just CPU==GPU agreement.

## 7. Where this sits in the real world

This is a **reduced-scope teaching version**: it isolates the induced-dipole SCF —
the defining, expensive kernel of AMOEBA — and omits the rest of the stack. A
production polarizable-MD code (the catalog's prior art) adds:

- **Permanent multipoles + Thole damping.** Real AMOEBA atoms carry charge,
  dipole, *and* quadrupole, with short-range interactions smeared (Thole) to
  prevent the polarization catastrophe. Our `b` is a clean fixed field instead.
- **PME / Ewald for long-range electrostatics.** For periodic systems the `O(N²)`
  all-pairs sum is replaced by **Particle-Mesh Ewald**: a real-space neighbor-list
  part plus a reciprocal-space part computed with an **FFT** (the catalog's cuFFT).
  This is the `O(N²) → O(N log N)` win that makes large systems tractable.
- **Preconditioned CG (PCG)** and extrapolation (e.g. ASPC) to cut SCF iterations,
  plus **analytic forces** (the gradient of `U_pol`) to drive the actual dynamics.
- **Scale and parallelism.** Tinker-HP and OpenMM-AMOEBA solve `3N` for `N` up to
  millions across multiple GPUs (**MPI domain decomposition + NCCL**), reporting
  >200× over single-CPU and enabling **microsecond** AMOEBA trajectories and
  AMOEBA-based binding free-energy (FEP) calculations.

What carries over exactly from this project: the **matrix-free CG structure**, the
**SPD induced-dipole operator**, the **warm-start**, the **FP64 + determinism**
discipline, and the GPU mapping idea. Those are the load-bearing concepts; the rest
is engineering scale on top of them.

---

## References

- **AMOEBA force field** — Ponder, Wu, Ren et al., *J. Phys. Chem. B* (2010): the
  polarizable multipole model and the induced-dipole SCF this project miniaturizes.
- **Tinker-HP** <https://github.com/TinkerTools/tinker-hp> — production GPU AMOEBA;
  study its (preconditioned) CG induced-dipole solver and PME-multipole.
- **OpenMM AMOEBA plugin** <https://github.com/openmm/openmm> — AMOEBA on CUDA; the
  reference multipole/induced-dipole math in readable form.
- **Tinker9** <https://github.com/TinkerTools/tinker9> — GPU-native rewrite; a
  clean look at data layout for polarizable MD on the GPU.
- **poltype2 / AMOEBA+** <https://github.com/TinkerTools/poltype2> — how the
  parameters this method consumes are generated.
- **Shewchuk, "An Introduction to the Conjugate Gradient Method Without the
  Agonizing Pain"** — the clearest derivation of the CG loop in §3.
- **NIST WebBook** <https://webbook.nist.gov> — thermophysical/dielectric data for
  validating a polarizable water model.
