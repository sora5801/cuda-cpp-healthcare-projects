# 1.23 — QM/MM Molecular Dynamics

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Drug%20Discovery%20%26%20Molecular%20Design-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 1: Drug Discovery & Molecular Design · Catalog ID `1.23`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

QM/MM molecular dynamics simulates a chemical reaction by treating the **reactive
atoms quantum-mechanically** (so bonds can break and form) while the surrounding
environment stays **classical** (cheap). This project is a **reduced-scope
teaching version** (CLAUDE.md §13): instead of a full DFT solve over hundreds of
atoms, the quantum "region" is a single proton on a **two-state model surface**,
and the classical environment enters as an **electrostatic-embedding field**.
Every MD step we (1) build a 2×2 quantum Hamiltonian, (2) diagonalize it *in
closed form* to get the ground-state energy and force, (3) add the classical MM
force, and (4) advance the proton one **velocity-Verlet** step. We then run a
whole **ensemble** of such trajectories — one GPU thread each — sweeping the MM
field and the starting position, and watch the proton stay trapped or transfer.

## What this computes & why the GPU helps

Hybrid quantum mechanics/molecular mechanics (QM/MM) partitions a system into a
reactive QM region (drug + key residues, 50–200 atoms) treated at DFT/semi-empirical
level and a larger MM region. GPU acceleration applies to both the QM Hamiltonian
(via TeraChem/GPU-DFT) and the MM dynamics (via AMBER/GROMACS). The critical
bottleneck is the QM/MM electrostatic coupling and QM Hamiltonian evaluation at
every MD step. Applications include enzyme catalysis mechanism, covalent drug
reactivity, and proton-transfer pathways.

**The parallel bottleneck (in this teaching version):** a single QM/MM trajectory
is *sequential* — each step needs the previous geometry to build the next quantum
Hamiltonian, so you cannot parallelize *within* one trajectory cheaply. What is
embarrassingly parallel is running **many independent trajectories** at once: a
sweep over MM field strengths and initial conditions (reactive-event sampling).
We map **one trajectory to one GPU thread**; each thread runs the full Verlet loop
(thousands of 2×2 quantum solves) in registers and writes one summary. This is the
**ensemble-integration** pattern (PATTERNS.md §1), shared with the 9.02 (SEIR) and
13.02 (PBPK) flagships.

## The algorithm in brief

- **Electrostatic embedding** — the classical MM field polarizes the QM
  Hamiltonian (the diagonal of a 2×2 matrix), tilting the reaction surface.
- **2-state QM "Hamiltonian"** — donor-bonded |L⟩ and acceptor-bonded |R⟩ states,
  coupled by an electronic tunneling element; **diagonalized analytically** (exact
  closed form for a symmetric 2×2 — no iterative solver, no library).
- **Born–Oppenheimer force** — the nuclear force is the analytic gradient of the
  ground-state (lower) eigenvalue; add the classical MM force.
- **Velocity Verlet** — the symplectic, time-reversible MD integrator ("Verlet MM"
  in the catalog) advances the proton; one quantum force evaluation per step.
- **Ensemble** — `nf × nx` independent trajectories, one GPU thread each.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/qm-mm-molecular-dynamics.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/qm-mm-molecular-dynamics.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\qm-mm-molecular-dynamics.sln /p:Configuration=Release /p:Platform=x64
```

The project links only `cudart_static.lib` — the quantum solve is an analytic 2×2
diagonalization, so **no cuSOLVER/cuBLAS is needed** (THEORY.md §4 explains why,
and when you *would* reach for cuSOLVER in the real version).

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (CMake build)
```

The demo builds if needed, runs on `data/sample/ensemble_params.txt`, prints the
result, shows the GPU-vs-CPU agreement check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/ensemble_params.txt` — one line defining the
  integration settings and a 16×16 `field × x0` sweep (256 trajectories). Offline,
  zero downloads.
- **Full dataset:** there is nothing to download — `scripts/download_data.ps1` /
  `.sh` print pointers to real enzyme/structure data (PDB, BRENDA, SAMPL) and the
  production GPU QM/MM engines you would graduate to.
- **Provenance & license:** see [data/README.md](data/README.md). **All data is
  synthetic**; the surface is a model, not a fitted quantum-chemistry PES.

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt). The
program integrates every trajectory on both the **GPU** (`src/kernels.cu`) and a
**CPU reference** (`src/reference_cpu.cpp`) — both calling the *same* shared
`__host__ __device__` core in `src/qmmm.h` — and asserts they agree to within
`1e-9` (the measured worst diff is ~`1e-12`). At `field = 0` the proton stays
**trapped** in the donor well; stronger embedding fields tilt the surface and the
proton **transfers**. On the committed sample, 90 / 256 trajectories transfer.

## Code tour

Read in this order:

1. [`src/qmmm.h`](src/qmmm.h) — **start here.** The shared `__host__ __device__`
   physics: the 2×2 QM solve, electrostatic embedding, the analytic force, and the
   velocity-Verlet step. This is the heart of the project.
2. [`src/main.cu`](src/main.cu) — loads the ensemble config, runs CPU + GPU,
   verifies, and prints the deterministic report (stdout) + timing (stderr).
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the one-thread-per-
   trajectory mapping idea.
4. [`src/kernels.cu`](src/kernels.cu) — the ensemble kernel and host wrapper.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline
   and the config loader.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

- **AMBER + QUICK** (<https://github.com/merzlab/QUICK>) — GPU-accelerated DFT
  engine for QM/MM with AMBER; study how a real QM region is solved each step.
- **TeraChem-TCPB** (<https://www.petachem.com>) — GPU DFT server driving QM/MM
  with NAMD/AMBER; learn the client/server split for the QM force.
- **OpenMM + PySCF QM/MM** (<https://github.com/openmm/openmm>) — a readable
  Python QM/MM interface; the cleanest place to see the embedding bookkeeping.
- **CP2K** (<https://github.com/cp2k/cp2k>) — GPU-accelerated QM/MM for periodic
  systems (Gaussian-and-plane-waves); the large-scale production view.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Ensemble integration — one independent trajectory per thread** (PATTERNS.md §1).
Each thread runs the entire velocity-Verlet time loop in registers and writes one
result struct; there is no shared memory, no atomics, and no inter-thread
communication. The per-step physics lives in a single `__host__ __device__` header
so the GPU and CPU compute byte-for-byte identical math (PATTERNS.md §2). Contrast
with the real version, which parallelizes *inside* one step (GPU DFT: electron-
repulsion integrals, density-matrix builds) — described in THEORY.md §7.

## Exercises

1. **Make it endergonic.** Flip the sign of the field range in
   `data/sample/ensemble_params.txt` (positive fields) and predict, then check,
   which way the proton transfers. Why is the surface symmetric at `field = 0`?
2. **Watch energy drift.** Add a step that prints total energy (kinetic + QM/MM
   potential) every 500 steps to stderr. Halve `dt` and confirm the drift shrinks
   ~4× (velocity Verlet is 2nd-order; see THEORY.md §5).
3. **Tune the barrier.** Change `COUPLING` in `src/qmmm.h` and observe how the
   barrier height (`≈ ½·KWELL·X_R² − COUPLING`) controls how easily the proton
   transfers. At what coupling does the double well disappear?
4. **Add a thermostat.** Replace the deterministic launch with per-thread cuRAND
   and a simple Langevin/Andersen thermostat to sample at finite temperature —
   then the ensemble estimates a *rate*, not just single trajectories.
5. **Profile occupancy.** This kernel is register-bound. Use Nsight Compute to
   read the register count, then try `__launch_bounds__` and block sizes 64/128/256
   and explain the occupancy curve.

## Limitations & honesty

- **Reduced scope (deliberate).** Real QM/MM solves the electronic structure of
  50–200 atoms with DFT or a semi-empirical method (GFN2-xTB) *every step*. Here
  the "QM region" is a **single proton on a 2×2 model Hamiltonian** with an exact
  closed-form solution. It teaches the QM/MM *loop* (build Hamiltonian → solve →
  force → Verlet → embedding) without the cost of a real solver. See THEORY.md §7.
- **Model units, synthetic surface.** The energies/lengths/masses are a
  self-consistent *model* system, **not** real atomic units, and the double-well
  surface is invented, not fitted to a molecule. Nothing here is a chemical or
  clinical prediction.
- **No link atoms, no real electrostatics.** A true QM/MM run caps cut covalent
  bonds with link atoms (ONIOM) and sums Coulomb interactions over thousands of MM
  point charges. We collapse all of that into one uniform embedding field.
- **Timing is a teaching artifact**, never a benchmark claim (CLAUDE.md §12).
