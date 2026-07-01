# 6.5 — Respiratory / Lung Airflow & Particle Deposition

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Computational%20Physiology%20%26%20Systems%20Biology-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 6: Computational Physiology & Systems Biology · Catalog ID `6.5`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

When you inhale a drug aerosol, *where its particles land* in the branching
airways decides whether the drug works. This project simulates that with
**Lagrangian particle tracking**: it follows hundreds of thousands of inhaled
aerosol particles, one at a time, down an idealized bifurcating airway tree, and
tallies which airway generation each one deposits in. Each particle is an
**independent** history, so the whole simulation maps beautifully onto the GPU —
**one thread per particle** — with deposition counts accumulated by deterministic
integer atomics. It is a compact, honest teaching version of the aerosol-dosimetry
models used in inhaled-drug and air-pollution research.

## What this computes & why the GPU helps

We track `N` inhaled particles through `n_gen` conducting-airway generations. In
each generation a particle can be removed from the airstream by three classical
mechanisms — **inertial impaction**, **gravitational sedimentation**, and
**Brownian diffusion** — whose combined probability we evaluate from the particle
size and the local airway radius, length, and air velocity. One random draw per
generation decides whether the particle deposits there or moves on. The outputs
are the **total deposition fraction** and the **per-generation deposition
profile**.

**The bottleneck that parallelizes:** every particle history is completely
independent (no particle affects any other). That is an embarrassingly parallel
workload — the catalog calls for "custom CUDA kernels for Lagrangian force
integration (one thread per particle) … with atomic-add deposition counters,"
which is exactly what we build. Millions of histories that would run serially on
the CPU run concurrently on the GPU.

## The algorithm in brief

- **Aerosol physics** (Stokes drag): relaxation time `τ`, settling velocity `v_s`,
  diffusion coefficient `D`, Cunningham slip correction `C_c`.
- **Per-generation deposition efficiencies**: impaction `η_imp(Stk)`,
  sedimentation `η_sed`, diffusion `η_diff`; combined survival
  `P = (1−η_imp)(1−η_sed)(1−η_diff)`.
- **Airway geometry**: a symmetric **Weibel-A** tree; continuity sets per-tube
  velocity `U[g] = Q / (2^g · π · r[g]²)`.
- **Monte-Carlo Lagrangian tracking**: one uniform draw per generation; deposit
  when `ξ ≥ P`; tally integer counts per generation.
- **GPU pattern**: one thread per particle, grid-stride loop, per-thread
  splitmix64 RNG, `atomicAdd` on 64-bit integer counters.

Full derivations and the GPU mapping are in **[THEORY.md](THEORY.md)**.

## Build

Prerequisites: **Visual Studio 2026** (v145 toolset, *Desktop development with
C++*) and **CUDA Toolkit 13.3** (see [`docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md)).

1. Open `build/respiratory-lung-airflow-particle-deposition.sln` in Visual Studio 2026.
2. Select the **`Release`** configuration and **`x64`** platform.
3. **Build → Build Solution** (`Ctrl+Shift+B`).

The executable lands in `build/x64/Release/respiratory-lung-airflow-particle-deposition.exe`.
Command line instead of the IDE:

```powershell
& "C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\MSBuild.exe" `
  build\respiratory-lung-airflow-particle-deposition.sln /p:Configuration=Release /p:Platform=x64
```

A cross-platform **CMake** build is also provided (see `CMakeLists.txt`); the VS
solution is the required deliverable.

## Run the demo

One command builds (if needed), runs on the committed sample, and checks the
result:

```powershell
./demo/run_demo.ps1        # Windows
```
```bash
./demo/run_demo.sh         # Linux/macOS (CMake path)
```

It prints the deposition profile (stdout), the timing (stderr), and `PASS`/`FAIL`.

## Data

The committed sample `data/sample/lung_params.txt` is a **synthetic**, human-
readable one-liner describing one experiment (5-µm particle, 30 L/min flow, 16
generations, 200 000 histories, fixed seed). It runs the demo **offline with zero
downloads**. The airway geometry is *built* deterministically from these numbers
(a scaled Weibel-A tree), so nothing else is needed.

Real studies replace the idealized tree with a geometry **segmented from a lung
CT scan** (LIDC-IDRI, COPDGene, SPIROMICS). Those archives require registration;
`scripts/download_data.*` prints the links and never bypasses credentials.
Regenerate or resize the synthetic sample with `python scripts/make_synthetic.py`.
Full provenance and per-field meanings are in [`data/README.md`](data/README.md).

## Expected output

```
6.5 -- Respiratory / Lung Airflow & Particle Deposition
aerosol: d_p = 5.0 um, rho_p = 1000 kg/m^3
airway : 16 generations, flow = 30.0 L/min
particles = 200000, seed = 12345
deposited = 43726 of 200000 (21.9%), exhaled = 156274
peak deposition generation = 0
deposition per generation (counts):
  7353 5910 4739 3922 3410 2725 2376 2048 1817 1646 1509 1395 1288 1232 1193 1163
RESULT: PASS (GPU deposition tally matches CPU exactly)
```

**How correctness is checked:** the program runs the computation on the CPU
(`reference_cpu.cpp`, serial, plain `++`) and on the GPU (`kernels.cu`, parallel,
`atomicAdd`) over the *identical* particle histories, and asserts the two
per-generation integer tallies are **exactly equal** (0 mismatches). Because the
tally is integer, atomic adds commute → the GPU result is deterministic and
matches the CPU bit-for-bit (see [THEORY.md](THEORY.md) "How we verify
correctness"). The full-precision timing goes to stderr and is a *teaching
artifact*, not a benchmark claim.

## Code tour

Read in this order:

1. **`src/main.cu`** — the 5-step shape: load → CPU reference → GPU → verify →
   report (deterministic stdout, timing on stderr).
2. **`src/lung_physics.h`** — the shared `__host__ __device__` core: aerosol
   physics, the three deposition efficiencies, the RNG, and `track_particle()`.
   *This is the heart of the project* and is compiled into both the CPU and GPU
   paths so their results match exactly.
3. **`src/kernels.cuh` → `src/kernels.cu`** — the GPU twin: one thread per
   particle, grid-stride loop, integer `atomicAdd` scoring.
4. **`src/reference_cpu.h` / `.cpp`** — the loader, the Weibel-A `build_airway()`,
   and the trusted serial reference.
5. **`src/util/`** — shared, heavily-commented `CUDA_CHECK`, CUDA-event timer, and
   host I/O helpers.

## Prior art & further reading

From the catalog's starter repos/tools (study these; do not copy — reimplement
didactically):

- **OpenFOAM `DPMFoam`** (https://github.com/OpenFOAM/OpenFOAM-dev) — production
  Lagrangian discrete-phase particle tracking (with GPU-accelerated AmgX pressure
  solves). Study how a full CFD code couples the carrier flow and the particles.
- **PALABOS** (https://gitlab.com/unigespc/palabos) — lattice-Boltzmann for
  alveolar-scale flow; the LBM alternative to finite-volume Navier–Stokes.
- **SimVascular** (https://github.com/SimVascular) — vascular CFD whose flow
  machinery adapts to airways.
- **3D Slicer + SlicerMorph** (https://github.com/SlicerMorph/SlicerMorph) — how a
  real airway geometry is segmented from a CT volume.
- **Aerosol/dosimetry background:** ICRP Publication 66 and the MPPD model —
  the deposition-probability formulas we implement here.

## Exercises

1. **Flip the dominant mechanism.** Rerun with `--d_p 0.01` (10 nm) via
   `make_synthetic.py`. Diffusion should now dominate and the deposition peak
   should move *deeper* into the tree. Explain why using the `Δ = D·t_res/r²`
   scaling.
2. **The U-curve.** Sweep `d_p` from 0.01 µm to 10 µm and plot total deposition
   fraction vs. size. You should recover the classic **U-shaped** curve (a minimum
   near ~0.3 µm). Add the sweep to a small script.
3. **Breathing effort.** Increase `flow_L_per_min` from 30 to 60 (exercise vs.
   rest). Which mechanism grows, and which shrinks? (Hint: impaction ∝ `U`,
   sedimentation/diffusion ∝ `t_res = L/U`.)
4. **Reduce warp divergence.** Sort particles by predicted deposition depth before
   launch, or compact still-alive particles between generations, and measure the
   effect on kernel time.
5. **Add a second aerosol.** Make the input polydisperse (a mix of sizes) and tally
   deposition per size bin — a step toward a realistic inhaler spectrum.

## Limitations & honesty

- **Reduced scope, on purpose.** This is a **deposition-probability** model on an
  **idealized symmetric tree**, not a CFD solve on a patient CT geometry. There is
  no resolved velocity field, no turbulence model, no gas exchange, and no
  particle–particle interaction. The full picture is described in
  [THEORY.md](THEORY.md) "Where this sits in the real world."
- **Synthetic data.** The committed sample and all demo numbers are synthetic and
  educational — **not** patient-derived and **not** clinically valid. Do not use
  this for any inhaler, drug-delivery, or exposure decision.
- **Semi-empirical formulas.** The efficiency correlations are standard textbook
  forms with representative constants; a real study would calibrate them (or
  replace them with CFD).
- **Timing is a teaching artifact.** The GPU/CPU times illustrate the pattern; they
  are not a benchmark. The GPU's edge grows with particle count.
