# 1.2 — Particle-Mesh Ewald Electrostatics

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Drug%20Discovery%20%26%20Molecular%20Design-lightgrey)

> **🟢 Beginner · Established** — Domain 1: Drug Discovery & Molecular Design · Catalog ID `1.2`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

Molecular-dynamics simulations of solvated proteins must compute the
electrostatic energy of thousands of charges under **periodic boundary
conditions** — where each charge also interacts with infinitely many periodic
images. That lattice sum does not converge if you naively truncate it.
**Particle-Mesh Ewald (PME)** is the standard fix: it splits the Coulomb sum into
a fast short-range part (`erfc`-damped, with a cutoff) and a smooth long-range
part evaluated on a 3D grid via an FFT, giving `O(N log N)` scaling. This project
computes PME's reciprocal-space energy on the GPU — **B-spline charge spreading
(an atomic scatter), a cuFFT 3D transform, and a reciprocal-space convolution** —
and verifies it three ways: against the same pipeline on the CPU, against the
textbook direct Ewald sum, and by confirming the total energy is invariant to the
Ewald splitting parameter β. It is a compact, honest tour of the single most
expensive kernel in modern MD.

## What this computes & why the GPU helps

Long-range electrostatics in periodic MD systems cannot be truncated without severe artifacts; PME splits the Coulomb sum into a short-range real-space part (evaluated with cutoff) and a smooth long-range reciprocal-space part evaluated on a 3D grid via FFT. The GPU acceleration opportunity is two-fold: the charge spreading (particle-to-mesh) and force interpolation (mesh-to-particle) steps are data-parallel over atoms, while the 3D FFT is handled by cuFFT. PME scales as O(N log N) and dominates walltime for large biological systems. Achieving double-precision accuracy at float throughput is the main engineering challenge.

**The parallel bottleneck:** the **reciprocal-space sum** dominates PME wall-time
for large systems. Its three sub-steps map cleanly onto the GPU: **charge
spreading** is one thread per atom scattering onto the grid (an atomic
scatter-add); the **3D FFT** is a single `cufftExecR2C` call (`O(K³ log K)`); and
the **reciprocal convolution** is one thread per grid bin. This project
accelerates exactly that pipeline; see [THEORY.md](THEORY.md) §4 for the
thread/block mapping.

## The algorithm in brief

- **Ewald split:** `E_total = E_real + E_recip − E_self`. The `erfc`-damped real
  part converges within a cutoff; the smooth reciprocal part is summed over
  wavevectors.
- **B-spline charge spreading (order 4):** each charge is interpolated onto a
  `4×4×4` block of grid points using cardinal B-splines (`src/pme.h`).
- **3D FFT (cuFFT R2C):** transforms the charge grid to the structure factor.
- **Reciprocal convolution:** multiply by the Ewald weight `exp(−k²/4β²)/k²` and
  the B-spline correction factor `B(m)`, then sum to get `E_recip`.
- **real-space `erfc` damping** and the **self-energy** correction complete the
  total; **smooth PME (SPME)** is the FFT-based scheme implemented here (the
  closely-related **P3M** is discussed in THEORY §7).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/particle-mesh-ewald-electrostatics.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/particle-mesh-ewald-electrostatics.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\particle-mesh-ewald-electrostatics.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/`, prints the result, shows the
GPU-vs-CPU agreement check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/` — a tiny, offline input so the demo runs
  with zero downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` (documented, idempotent).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: CHARMM-GUI solvation benchmark sets — pre-built periodic protein-water boxes (https://charmm-gui.org); D. E. Shaw Research Anton trajectories — ms-scale trajectory archives (available via DE Shaw); ion channel benchmark systems (MemProtMD, https://memprotmd.bioch.ox.ac.uk).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt):

```
1.2 -- Particle-Mesh Ewald Electrostatics
system: 64 charges in a cubic box of side 8.0000 (reduced units)
PME params: grid 16x16x16, B-spline order 4, beta 0.750000, rcut 4.000000
E_recip (GPU SPME) = 2.02433442
E_recip (CPU SPME) = 2.02433454
E_recip (direct Ewald) = 2.02433454
E_real = -2.90499951   E_self = 27.08110001
E_total (real + recip - self) = -27.96176498
CHECK GPU==CPU SPME      : PASS (rel 6.15e-08 <= 1e-04)
CHECK SPME~=direct Ewald : PASS (rel 7.46e-12 <= 5e-03)
CHECK total invariant to beta : PASS (rel 2.62e-05 <= 2e-02)
RESULT: PASS
```

The program computes `E_recip` on the **GPU** (`src/kernels.cu`), on the **CPU**
via the identical SPME pipeline, and via the **direct Ewald** k-sum
(`src/reference_cpu.cpp`), and checks all three plus the β-invariance of the total
energy. That triple agreement is the correctness guarantee. (Timings and the FP32
error go to **stderr**, which the demo shows but does not diff.)

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads the system, runs CPU + GPU, verifies (3
   checks), reports.
2. [`src/pme.h`](src/pme.h) — the shared **host+device** physics: B-spline
   weights, the fixed-point charge encoding, and the B-spline modulus. Read this
   before the kernels — it is the math both sides share.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU pipeline interface + the
   spread→FFT→convolve→reduce idea.
4. [`src/kernels.cu`](src/kernels.cu) — the atomic spreading kernel, the cuFFT
   call (documented, not a black box), and the energy kernel.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baselines
   (SPME twin + direct Ewald + real/self terms + the separable host DFT).
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

GROMACS CUDA PME (https://github.com/gromacs/gromacs) — reference GPU PME implementation; NAMD GPU PME (https://www.ks.uiuc.edu/Research/namd/) — tiled domain-decomposed PME; OpenMM PME plugin (https://github.com/openmm/openmm) — Python-accessible PME with mixed-precision; cuFFT (https://developer.nvidia.com/cufft) — NVIDIA's FFT library used internally by all above.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**cuFFT for the 3D FFT** + **an atomic, fixed-point scatter for charge spreading**.
This project combines two flagship patterns from the cookbook
([docs/PATTERNS.md](../../../docs/PATTERNS.md)): *using a library without a black
box* (cuFFT R2C, like `8.03`) and *parallel scatter + atomic reduction made
deterministic with fixed-point integers* (like `11.09`). Charge spreading is one
thread per atom doing `atomicAdd` onto a `K³` integer grid; the FFT is one
`cufftExecR2C`; the convolution is one thread per reciprocal bin; the final energy
is summed on the host in fixed order so stdout is reproducible.

## Exercises

1. **Forces (mesh→particle).** The big omission: real MD needs the *force* on each
   atom, the gradient of `E_recip`. Add a second interpolation that gathers the
   convolved potential back onto atoms using the **derivatives** of the same
   B-spline weights. Verify against a finite-difference of the energy.
2. **Spline order 6.** Raise `PME_ORDER` to 6 and watch the SPME-vs-direct error
   shrink at fixed `K`. Plot error vs. `K` for orders 4 and 6.
3. **Shared-memory spreading.** The current `spread_kernel` does plain global
   `atomicAdd`. Tile atoms into shared memory and accumulate a block-local grid
   patch first to cut atomic contention; compare timings.
4. **A β / K sweep.** For a fixed accuracy target, find the `(β, K)` that minimizes
   total (real + reciprocal) cost — the real tuning problem production codes solve.
5. **Scale it up.** Generate a 512-ion system (`make_synthetic.py --reps 8
   --box 16`) and watch the cuFFT-vs-host-DFT gap widen as `K` grows.

## Limitations & honesty

- **Energy only, no forces.** Production PME computes forces (the gradient); we
  compute the reciprocal *energy* and leave forces as Exercise 1.
- **Synthetic data.** The sample is a **synthetic** NaCl-like ionic lattice
  (labeled synthetic everywhere), not a real protein. It is chosen because its
  energy is a clean, recognizable target — not because it is biologically
  realistic.
- **Reduced units.** Coulomb's constant is 1, so energies are in reduced units, not
  kcal/mol. A real code multiplies by the appropriate physical constant.
- **FP32 FFT.** cuFFT here is single precision, so the GPU energy differs from the
  FP64 host value at ~`10⁻⁷` relative (documented; the verification tolerates it).
  The displayed low-order GPU digits can differ on a different GPU.
- **`O(N²)` real space, naive host DFT.** The teaching real-space sum is `O(N²)`
  (no neighbour list) and the host reference DFT is `O(K⁴)` (no FFTW). Both are
  fine for the tiny sample and deliberately transparent; production uses neighbour
  lists and FFT libraries.
- **Not for clinical use.** This is study material illustrating an algorithm, not a
  validated simulation tool.
