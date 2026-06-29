# 1.21 — Polarizable / AMOEBA Force Field MD

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Drug%20Discovery%20%26%20Molecular%20Design-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 1: Drug Discovery & Molecular Design · Catalog ID `1.21`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

The thing that makes the **AMOEBA** force field "polarizable" — and ~10× more
expensive than a fixed-charge field like AMBER — is that every atom carries an
**induced dipole** that responds to the electric field of all the other atoms,
which in turn depend on it. Resolving this chicken-and-egg coupling means solving
a **self-consistent linear system** `A μ = b` at *every* molecular-dynamics step.
This project implements the standard solution — a hand-written, matrix-free
**conjugate-gradient (CG)** solver for the induced dipoles — and runs an
**ensemble** of these solves in parallel on the GPU, **one CUDA thread per
system**. It is a faithful, didactic miniature of the kernel at the heart of
Tinker-HP / OpenMM-AMOEBA.

## What this computes & why the GPU helps

Classical fixed-charge force fields miss polarization effects crucial for accurate
binding free energies and ionic interactions. The AMOEBA force field includes
point multipoles (up to quadrupoles) and induced dipoles solved self-consistently
at each MD step via an iterative solver (conjugate gradient). This increases cost
~10× over AMBER but GPU implementation in Tinker-HP achieves >200-fold speedup
over single-CPU, making microsecond AMOEBA simulations of large proteins feasible.
Applications include protein-ligand FEP with AMOEBA and pKa prediction in complex
electrostatic environments.

**The parallel bottleneck:** the induced-dipole **self-consistent field (SCF)**
solve. Each MD step must solve `A μ = b` (a 3N-dimensional symmetric
positive-definite system) to convergence — the dominant cost of an AMOEBA step.
Here we expose parallelism the simplest legible way: many **independent** systems
(an ensemble of configurations, as in a sweep / Monte-Carlo / per-step batch),
each solved by its own GPU thread. Within one large system, the parallelism lives
instead in the `O(N²)` matrix-vector product and the dot-product reductions — the
block-per-system design we describe in [THEORY.md](THEORY.md) and leave as an
exercise.

## The algorithm in brief

- **Induced-dipole equation** `μ_i = α_i (E_i^perm + Σ_{j≠i} T_ij · μ_j)`, recast
  as a symmetric positive-definite linear system `A μ = b`.
- **Matrix-free conjugate gradient** — the canonical Krylov solver for SPD
  systems; never forms `A`, only multiplies by it (the dipole-field "matvec").
- **Dipole–dipole interaction tensor** `T_ij = (3 r r^T − r² I)/r⁵` — the
  off-diagonal coupling.
- **Ensemble mapping** — one independent CG solve per GPU thread (the catalog's
  "custom CUDA conjugate-gradient solver for induced dipoles").
- *(Named in the catalog, discussed in THEORY, out of scope for this teaching
  version: PME-multipole Ewald, the AMOEBA water model, HIPPO, PIMD.)*

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/polarizable-amoeba-force-field-md.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/polarizable-amoeba-force-field-md.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\polarizable-amoeba-force-field-md.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/`, prints the per-member induced
dipoles + polarization energies, shows the GPU-vs-CPU agreement check, and prints
a timing line.

## Data

- **Sample (committed):** `data/sample/amoeba_ensemble.txt` — a tiny, **synthetic**
  ensemble of 8 polarization systems so the demo runs with zero downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` (print pointers to real
  AMOEBA parameter sets and validation data; never bypass credentials).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: AMOEBA protein force field parameter files
(<https://github.com/TinkerTools/tinker>); WaterMap/hydration site datasets
(Schrödinger, verify URL); BindingDB experimental affinities
(<https://www.bindingdb.org>); NIST thermophysical properties
(<https://webbook.nist.gov>).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt). The
program solves the induced dipoles on both the **GPU** (`src/kernels.cu`) and a
**CPU reference** (`src/reference_cpu.cpp`) and asserts they agree within `1.0e-9`
— that agreement is the correctness guarantee. Because both paths run the
*identical* double-precision CG loop (shared via `src/amoeba.h`), the actual worst
per-member difference is ~`1e-17` (machine precision), printed on stderr.

The per-member table shows the polarization energy and induced dipoles **growing**
as the two partner atoms approach (half-sep 4.0 → 2.0 Å) — the qualitative
signature of a polarizable model.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads/synthesizes the ensemble, runs CPU + GPU,
   verifies, reports.
2. [`src/amoeba.h`](src/amoeba.h) — **the heart**: the shared `__host__ __device__`
   physics — the dipole-field operator `apply_A` and the matrix-free CG solver
   `solve_induced_dipoles`. CPU and GPU call the *same* code.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the thread-per-system
   mapping idea.
4. [`src/kernels.cu`](src/kernels.cu) — the kernel and host wrapper (allocate,
   copy, launch + time, copy back).
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the loader, the synthetic
   builder, and the serial CG baseline.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, the CUDA-event timer, I/O.

## Prior art & further reading

Tinker-HP (<https://github.com/TinkerTools/tinker-hp>) — massively parallel GPU
AMOEBA MD; the production reference for induced-dipole CG/PCG on the GPU.
OpenMM AMOEBA plugin (<https://github.com/openmm/openmm>) — AMOEBA on CUDA; study
its `AmoebaReferenceMultipoleForce` for the multipole/induced-dipole math.
Tinker9 (<https://github.com/TinkerTools/tinker9>) — GPU-native Tinker rewrite.
AMOEBA+ parameters (<https://github.com/TinkerTools/poltype2>) — how the
parameters this method consumes are generated.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2). Our CG is written
from scratch precisely so nothing is a black box.

## CUDA pattern used here

**Ensemble of independent solves — one thread per system** (PATTERNS.md "ensemble"
row, exemplified by flagship 9.02). Each thread runs the full matrix-free
conjugate-gradient loop in registers/local memory and writes one result; no shared
memory, no atomics, no cross-thread communication. This realizes the catalog's
"custom CUDA conjugate-gradient solver for induced dipoles." The catalog also lists
cuFFT (multipole PME), warp-synchronous energy reductions, and multi-GPU via
MPI/NCCL — those belong to the full production stack and are described, not built,
in [THEORY.md](THEORY.md) ("Where this sits in the real world").

## Exercises

1. **Watch CG converge.** The weak synthetic coupling converges in ~2 iterations.
   Increase `alpha` (e.g. 6–8 Å³) or shrink the separations in
   `make_synthetic.py`, and observe the iteration count climb — and eventually
   diverge (the "polarization catastrophe"). Add a per-iteration residual print to
   stderr to see the geometric decay.
2. **Add a preconditioner.** Replace plain CG with **diagonally preconditioned**
   CG (PCG): scale the residual by `α_i` each iteration. Production AMOEBA codes
   use exactly this. Measure the drop in iteration count.
3. **Block-per-system.** Re-map so a *thread block* cooperates on one large system:
   parallelize the `O(N²)` matvec across threads and use a block reduction
   (`__syncthreads` + shared memory, or `cub::BlockReduce`) for the CG dot
   products. Compare against thread-per-system as `N` grows.
4. **A second verification metric.** For a single isolated atom, the exact answer
   is `μ = α E` with zero coupling; add an assertion against this closed form.
5. **FP32 vs FP64.** Switch the CG arrays to `float` and watch the verification
   tolerance you need grow — a concrete lesson in why polarization solves like
   double precision.

## Limitations & honesty

- **Reduced-scope teaching model.** This implements the *induced-dipole SCF* — the
  defining, expensive piece of AMOEBA — and nothing else. It omits permanent
  multipoles beyond a fixed driving field, Thole damping of short-range
  interactions, periodic boundary conditions / **PME**, real AMOEBA parameters,
  and the surrounding MD integrator. The `O(N²)` all-pairs matvec replaces the
  neighbor-list + PME that production codes use; it is correct but does not scale.
- **Synthetic data, illustrative units.** Geometry, field, and polarizabilities are
  chosen for didactic clarity, not chemical accuracy, and are in reduced units.
  Labeled synthetic everywhere (see [data/README.md](data/README.md)).
- **Timing is a teaching artifact, not a benchmark.** On a tiny 8-system ensemble
  the GPU is *slower* than the CPU because the run is dominated by kernel-launch
  and copy overhead; the GPU's advantage appears only with many systems (or many
  atoms). This is stated plainly and never dressed up as a speed-up claim.
- **Not for any real-world or clinical use.** Study material only.
