# 2.34 — Biophysical Simulation of Biomolecular Condensates (Active Learning Loop)

![difficulty](https://img.shields.io/badge/difficulty-Advanced-blue) ![maturity](https://img.shields.io/badge/maturity-Frontier%2FTheoretical-informational) ![domain](https://img.shields.io/badge/domain-Structural%20Biology%20%26%20Protein%20Science-lightgrey)

> **🔴 Advanced · Frontier/Theoretical** — Domain 2: Structural Biology & Protein Science · Catalog ID `2.34`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

Biomolecular condensates are membraneless droplets that form when disordered
proteins phase-separate; their **compactness** and **internal mobility** depend on
the protein sequence. Discovering which sequence gives a target property is an
**active-learning loop**: simulate many candidate sequences, learn the property
landscape, propose the next sequence. This project builds a **reduced-scope
teaching version** of that loop's GPU core: each candidate sequence (reduced to a
single "stickiness" `lambda`) gets its own GPU thread, which runs a full
coarse-grained Brownian-dynamics trajectory and measures a radius of gyration
`Rg` and a diffusion coefficient `D`; a deterministic acquisition step then
proposes the candidate whose `D` best matches an experimental target. It is a
clean, verifiable instance of the **"one thread integrates one trajectory"**
ensemble pattern.

## What this computes & why the GPU helps

Understanding the sequence determinants of biomolecular condensate properties
(surface tension, viscosity, partition coefficients of client molecules) requires
an active-learning loop: GPU CG-MD generates condensate properties, a surrogate
learns the property landscape, and Bayesian optimization proposes new sequences.
This closes the loop between sequence, structure, and function for disordered
proteins. GPU acceleration enables the throughput (hundreds of condensate
simulations per iteration). Applications include designing condensate-targeting
therapeutics and understanding IDR evolution.

**The parallel bottleneck:** the loop's cost is **simulating the ensemble** —
every candidate sequence needs its own independent CG-MD trajectory. A trajectory
is sequential *in time* but independent of the other candidates, so the ensemble
is embarrassingly parallel **across members**: each GPU thread integrates one
candidate's entire trajectory in its own registers/local memory, with no
inter-thread communication. That is the step the GPU parallelizes and the step
that dominates a real iteration's runtime.

## The algorithm in brief

- **Per candidate (one GPU thread):** overdamped Langevin / Brownian dynamics
  (Euler–Maruyama) of a bead-spring chain with harmonic bonds + a cohesive well
  whose stiffness scales with the sequence stickiness `lambda`.
- **Deterministic thermal noise:** a counter-based hash RNG keyed on
  `(replica, step, bead, axis)` so CPU and GPU draw identical kicks.
- **Observables:** time-averaged radius of gyration `Rg`; internal mobility `D`
  from a lag-MSD in the centre-of-mass frame via the Einstein relation.
- **Active-learning step:** propose the candidate whose `D` is closest to the
  experimental `target_D` (deterministic argmin of `|D − target_D|`).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation, and §0 there for exactly what is reduced relative to the frontier
project.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/biophysical-simulation-of-biomolecular-condensates-active-learning-loop.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/biophysical-simulation-of-biomolecular-condensates-active-learning-loop.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\biophysical-simulation-of-biomolecular-condensates-active-learning-loop.sln /p:Configuration=Release /p:Platform=x64
```

Both `Release|x64` and `Debug|x64` build with zero warnings; they produce
byte-identical stdout (the dynamics are fully deterministic).

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/condensate_ensemble.txt`, prints
the ensemble table and the active-learning proposal, shows the GPU-vs-CPU
agreement check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/condensate_ensemble.txt` — a tiny, offline
  **synthetic** experiment configuration so the demo runs with zero downloads.
- **Regenerate / scale up:** `python scripts/make_synthetic.py [--n-members N]`.
- **"Full dataset" pointers:** `scripts/download_data.ps1` / `.sh` (print links to
  PhaSePro, DisProt, PDB, CALVADOS; never bypass terms of use).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: PhaSePro (https://phasepro.elte.hu); DisProt
(https://disprot.org); experimental LLPS partition coefficient datasets (verify
URL); published condensate MD trajectory datasets (FUS, TDP-43, hnRNPA1).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt): a
five-row ensemble table (`lambda → Rg  D  |D-target|`) followed by the
active-learning proposal and `RESULT: PASS`. The program computes the ensemble on
both the **GPU** (`src/kernels.cu`) and a **CPU reference**
(`src/reference_cpu.cpp`) and asserts they agree within the documented tolerance
(`1e-6`; observed `~1.3e-15`) — that agreement is the correctness guarantee. The
physical sanity check is built into the sample: `Rg` falls monotonically with
`lambda`, and the proposal recovers the known interior optimum (member `m12`).

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads the config, runs CPU + GPU, verifies, and
   prints the ensemble table + the active-learning proposal.
2. [`src/condensate.h`](src/condensate.h) — the shared `__host__ __device__`
   physics: the counter-based RNG, the forces, and `integrate_replica` (the whole
   trajectory). **The heart of the project.**
3. [`src/kernels.cuh`](src/kernels.cuh) → [`src/kernels.cu`](src/kernels.cu) — the
   GPU interface and the one-thread-per-trajectory kernel + host wrapper.
4. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline
   and the deterministic active-learning acquisition.
5. [`src/util/`](src/util/) — shared `CUDA_CHECK`, the CUDA-event timer, host I/O.

## Prior art & further reading

- **CALVADOS 2** (https://github.com/KULL-Centre/CALVADOS) — residue-level IDP CG
  model; learn how a real per-residue stickiness (`lambda_a`) table replaces our
  single scalar.
- **OpenMM** (https://github.com/openmm/openmm) — GPU MD engine you would drive the
  CG-MD with; study its Langevin integrators.
- **LAMMPS GPU** (https://github.com/lammps/lammps) — large-scale CG condensate
  (slab) simulation for coexistence/surface-tension; the next rung up in fidelity.
- **BoTorch** (https://github.com/pytorch/botorch) — GPU Bayesian optimization;
  see how an acquisition function with uncertainty replaces our argmin.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Ensemble integration — one thread integrates one full trajectory** (PATTERNS.md
§1, the flagship-9.02 / 13.02 pattern). Each thread runs a candidate sequence's
entire Brownian-dynamics loop in registers/local memory, with a deterministic
counter-based RNG for reproducibility, no shared memory, and no atomics; the only
host-side reduction is the active-learning argmin. The original catalog note
(GNN surrogate + BoTorch + multi-GPU) is the frontier version this teaching
project reduces — see [THEORY.md](THEORY.md) §0 and §7.

## Exercises

1. **Bigger ensemble, real speed-up.** Run
   `python scripts/make_synthetic.py --n-members 400` and rebuild. Watch the GPU
   time stay roughly flat while the CPU time grows — at what ensemble size does the
   GPU overtake the CPU on your card?
2. **Sharper diffusion estimate.** `D` is noisy from a single short trajectory.
   Average MSD over *several* lags (a small `tau` sweep, fit `MSD ∝ tau`) inside
   `integrate_replica`, and check whether `D(lambda)` becomes monotone.
3. **Two-knob sequence space.** Add a second sequence descriptor (e.g. a charge
   parameter that adds a long-range term) and turn the 1-D scan into a 2-D grid;
   make the proposal an argmin over both.
4. **Real acquisition.** Replace `acquisition_score` with Expected Improvement over
   a tiny Gaussian-process surrogate fit to the swept points — the first real step
   toward Bayesian optimization.
5. **Shared-memory experiment.** The trajectory is local-memory-bound. Try caching
   the ring buffer differently (or shrinking `lag`) and measure the effect on the
   kernel time.

## Limitations & honesty

- **Reduced scope.** This is a teaching reduction (CLAUDE.md §13): one scalar
  `lambda` instead of a per-residue force field, a harmonic cohesion well instead
  of Ashbaugh–Hatch + electrostatics, single-chain Rg/D instead of slab-based
  coexistence/surface-tension, and a deterministic argmin instead of a GNN
  surrogate + Bayesian optimization. THEORY §7 maps each simplification to its
  production counterpart.
- **Synthetic data.** The input is a synthetic configuration, labeled synthetic
  everywhere. The reported `Rg` and `D` are outputs of a toy model, **not**
  quantitative predictions for FUS/TDP-43/hnRNPA1 or any real protein.
- **Noisy `D`.** A single short trajectory gives a noisy mobility estimate; the
  downward `D(lambda)` trend is real but scattered (Exercise 2 fixes it). `Rg` is
  the clean monotone observable.
- **Timing is a teaching artifact**, not a benchmark claim; on the tiny committed
  ensemble the GPU is intentionally *slower* than the CPU (PATTERNS.md §7).
- **No clinical meaning.** Nothing here may inform diagnosis or treatment.
