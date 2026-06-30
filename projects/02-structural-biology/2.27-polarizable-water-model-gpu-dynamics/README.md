# 2.27 — Polarizable Water Model GPU Dynamics

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Structural%20Biology%20%26%20Protein%20Science-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 2: Structural Biology & Protein Science · Catalog ID `2.27`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

Polarizable water models (AMOEBA, SWM4-NDP/Drude, and the gold-standard **MB-pol**)
improve on fixed-charge water (TIP3P) by letting each molecule's dipole *respond*
to its surroundings — essential for dielectric constants, ion solvation, and
protein hydration. The expensive, iterative heart they all share is the
**self-consistent induced-dipole solve**: every polarizable site carries a dipole
`µ = αE` where the field `E` comes from all the *other* dipoles, so the dipoles
must be found together by iteration. This project solves that fixed point with a
**Jacobi self-consistent-field (SCF)** loop on the GPU — one thread per site per
sweep, ping-ponging two dipole buffers — and verifies it against an identical CPU
reference and an analytic single-dipole answer. It is a faithful, heavily
commented, reduced-scope window into what GPU polarizable-MD codes do inside their
solvers.

## What this computes & why the GPU helps

Accurate water models are foundational to all biomolecular simulation. The MB-pol
many-body water potential, TIP4P-D, and OPC3/OPC models improve upon TIP3P for
protein hydration dynamics, but many-body polarizable water (MB-pol) is orders of
magnitude more expensive due to its 2-body and 3-body interaction terms. GPU
acceleration (via MB-nrg and the **MBX** library) is what makes multi-nanosecond
polarizable production runs feasible. Applications include protein solvation
thermodynamics and dielectric-constant convergence for force-field validation.

**The parallel bottleneck:** the **induced-dipole SCF**. Each self-consistent
sweep evaluates, for every site, the electric field from *all other* induced
dipoles — an `O(N²)` all-pairs computation — and the sweep is repeated until the
dipoles stop changing. Within a sweep the N site updates are independent, so the
GPU gives **one thread per site**: each thread does the `O(N)` field gather for its
own site, and all N proceed in parallel. This is the step that dominates
polarizable-MD runtime and the one GPUs accelerate. See
[THEORY.md §4](THEORY.md) for the full mapping.

## The algorithm in brief

- **Inducible point dipoles:** `µ_i = α_i E_i` (the polarizable degree of freedom).
- **Self-consistent field:** `E_i` includes the permanent charges *and* every other
  induced dipole, so the dipoles are coupled and solved iteratively.
- **Jacobi (fixed-point) iteration:** update every site from the previous sweep's
  dipoles, swap buffers, repeat until the max dipole change < tol — the parallel-
  friendly cousin of Gauss–Seidel.
- **Thole damping** of the dipole field tensor to avoid the short-range
  *polarization catastrophe*.
- **Induction energy** `U_pol = −½ Σ µ_i · E_i^perm` as the headline scalar result.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation, including the conjugate-gradient and MB-pol extensions the catalog
mentions.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/polarizable-water-model-gpu-dynamics.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/polarizable-water-model-gpu-dynamics.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\polarizable-water-model-gpu-dynamics.sln /p:Configuration=Release /p:Platform=x64
```

This project links only the CUDA runtime (`cudart_static.lib`) — no extra CUDA
libraries — so the build is self-contained.

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (CMake build)
```

The demo builds if needed, runs on `data/sample/water_cluster.txt`, prints the
converged dipoles + polarization energy, shows the GPU-vs-CPU agreement check and
the analytic probe check, and prints a timing line on stderr.

## Data

- **Sample (committed):** `data/sample/water_cluster.txt` — a tiny **synthetic**
  cluster (an isolated polarizable probe + two water-like molecules) so the demo
  runs offline with zero downloads. It is also the program's built-in fallback.
- **Generator:** `scripts/make_synthetic.py` reproduces the sample byte-for-byte
  and can make larger clusters (`--waters 64`).
- **Reference data pointers:** `scripts/download_data.ps1` / `.sh` print where the
  real-world data lives (no download is needed).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: NIST water thermophysical properties
(<https://webbook.nist.gov>); HBond-dynamics NMR benchmarks; MD2PDB water
trajectory archives; SPC/E, TIP4P-2005 reference simulation datasets.

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt):

```
2.27 -- Polarizable Water Model GPU Dynamics
self-consistent induced dipoles (Jacobi SCF) on 7 sites
a_thole = 0.390  tol = 1.0e-09  max_iters = 200
converged in 10 sweeps
induced dipole magnitude |mu| per site (e*A):
  site  0: q=+0.000 alpha=1.444  |mu|=0.072198272
  site  1: q=-0.834 alpha=1.444  |mu|=0.810270988
  ...
polarization energy U_pol = -0.591838205 e^2/A = -196.527992 kcal/mol
probe check: |mu0| = 0.072198272  analytic alpha*Eext = 0.072200000
RESULT: PASS (GPU matches CPU within tol=1.0e-09)
```

The result is computed on both the **GPU** (`src/kernels.cu`) and a **CPU
reference** (`src/reference_cpu.cpp`); they agree to ≤1e-9 (round-off, since both
run identical FP64 arithmetic and reduce in fixed point). The **probe check**
recovers the analytic `µ = αE` independently, validating the physics. That double
agreement is the correctness guarantee.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads the cluster, runs CPU + GPU, verifies
   (dipoles, energy, and the analytic probe), reports.
2. [`src/polar.h`](src/polar.h) — the **shared `__host__ __device__` physics**: the
   permanent field, the Thole-damped dipole field tensor, the induction energy.
   This is the one source of truth both CPU and GPU call.
3. [`src/reference_cpu.h`](src/reference_cpu.h) / [`reference_cpu.cpp`](src/reference_cpu.cpp)
   — the system config, the loader, and the trusted serial Jacobi SCF.
4. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the thread-mapping idea.
5. [`src/kernels.cu`](src/kernels.cu) — the four kernels and the host SCF loop.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

- **MBX** (<https://github.com/paesanilab/MBX>) — GPU-accelerated many-body water
  (MB-pol); study how it splits 1-/2-/3-body terms and offloads them to the GPU.
- **OpenMM** (<https://github.com/openmm/openmm>) — production GPU inducible-dipole
  (AMOEBA/Drude) water; see its mutual-induced-dipole CG solver.
- **Tinker-HP** (<https://github.com/TinkerTools/tinker-hp>) — large-scale
  polarizable AMOEBA MD; a reference for PME + polarization on GPUs.
- **i-PI** (<https://github.com/i-pi/i-pi>) — a driver for path-integral water
  dynamics; useful for seeing how a force engine plugs into an MD/PIMD loop.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Iterative relaxation + N-body field evaluation** (PATTERNS.md): one thread per
site, an `O(N)` all-pairs field gather inside each thread, a host-driven Jacobi
sweep loop, and **ping-pong double buffering** of the dipole arrays. Closest
flagship exemplars: **10.02** (PBD — Jacobi projection with a double buffer) for
the iterate-to-self-consistency structure, and **9.02/13.02** (ensemble RK4) for
the shared `__host__ __device__` core idiom. Scalar reductions (max residual, total
energy) use **fixed-point integer atomics** for determinism (cf. **5.01**, **11.09**).
The catalog's full vision — a GPU **conjugate-gradient** dipole solver, cuBLAS
many-body tensor contractions, and batched neighbour lists — is described as the
real-world extension in [THEORY.md §8](THEORY.md).

## Exercises

1. **Break it on purpose.** Set `a_thole` to `0` in `data/sample/water_cluster.txt`
   and move the two waters to ~1.0 Å apart. Watch the SCF diverge (the polarization
   catastrophe), then restore the damping and see it converge again.
2. **Gauss–Seidel.** Modify the CPU reference to update `µ_i` in place (using the
   already-updated `µ_{<i}`). Count how many fewer sweeps it needs — and explain
   why that version cannot be parallelized the way the GPU sweep is.
3. **Scale it.** Generate `--waters 256` with `make_synthetic.py` and watch the
   GPU's time advantage grow as `N` rises (timing is on stderr).
4. **Conjugate gradient.** Replace the Jacobi loop with a (matrix-free) CG solve of
   `Aµ = E^perm`, reusing `dipole_sweep_kernel`'s field evaluation as the matvec.
   Compare sweep counts.
5. **Shared-memory tiling.** Tile the inner `j` loop of `dipole_sweep_kernel`
   through shared memory (cooperatively load a block of `µ_j`) and measure the
   speed-up on a larger cluster.

## Limitations & honesty

- **Reduced-scope teaching version** (CLAUDE.md §13). It implements the
  inducible-dipole SCF — the shared core of polarizable models — *not* the full
  MB-pol many-body (2-body dispersion + 3-body induction) potential, and not an MD
  time loop. Those are described, not coded (THEORY.md §8).
- **Jacobi, not conjugate gradient.** Production codes use preconditioned CG (fewer
  iterations); Jacobi is chosen for transparency and parallelism.
- **`O(N²)` all-pairs, no cutoff/PME.** Fine for the tiny demo cluster; a real box
  needs neighbour lists + Particle-Mesh Ewald for the long-range tail.
- **Synthetic data.** The cluster, charges (TIP-style), and polarizability
  (water's molecular `α = 1.444 Å³`) are a simplified model labelled synthetic
  everywhere — not a fitted force field and not measured coordinates.
- **No clinical or research claim.** Energies and dipoles here are a software
  demonstration of the solver, not a validated water-model result.
