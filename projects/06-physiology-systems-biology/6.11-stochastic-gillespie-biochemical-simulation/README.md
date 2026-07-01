# 6.11 — Stochastic (Gillespie) Biochemical Simulation

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Computational%20Physiology%20%26%20Systems%20Biology-lightgrey)

> **🟢 Beginner · Established** — Domain 6: Computational Physiology & Systems Biology · Catalog ID `6.11`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

When a cell contains only a handful of copies of a molecule — a few mRNAs, a
dozen transcription factors — chemistry is *noisy*: reactions fire one at a time,
at random moments, and the copy number jitters around its average. The smooth
deterministic ODE (mass-action kinetics) is simply the wrong model in that
regime. The **Gillespie Stochastic Simulation Algorithm (SSA)** instead produces
*exact* random realizations of the underlying **Chemical Master Equation (CME)**:
every reaction event happens at a correctly-distributed random time. To turn
those noisy single runs into statistics (means, variances, distributions) you
must simulate **many independent trajectories** and average. Each trajectory is
completely independent, so this project runs **one trajectory per GPU thread** —
the cleanest possible parallel Monte-Carlo pattern — and verifies the GPU
ensemble against a byte-identical CPU reference and against a closed-form
analytic answer.

## What this computes & why the GPU helps

The Gillespie SSA exactly samples the master equation for discrete molecular
counts in a well-mixed chemical reaction network — critical when molecule numbers
are small (transcription factors, signaling molecules). Each stochastic
trajectory is independent, so GPU parallelism maps one trajectory per thread.
With 1,000–10,000 trajectories needed for statistics, GPU batch SSA achieves
large speed-ups. Tau-leaping approximations (binomial/Poisson) trade exactness
for speed at higher copy numbers.

**The parallel bottleneck:** the cost is the *ensemble* — you need thousands of
independent SSA runs, each an event-by-event loop over the reaction network.
A single trajectory is inherently serial in time (event *n+1* depends on the
state after event *n*), but the trajectories do not depend on one another. So the
GPU parallelizes **across trajectories**: thread `idx` runs the entire time loop
for trajectory `idx` in its own registers, with its own random-number stream, and
never talks to any other thread — no atomics, no shared memory, no
synchronization.

## The algorithm in brief

- **Gillespie SSA, direct method** (implemented here): at each step compute every
  reaction's mass-action *propensity* `a_j`, draw the waiting time to the next
  event as `τ = -ln(u₁)/a₀` (exponential with total rate `a₀ = Σⱼ aⱼ`), pick which
  reaction fires with probability `aₖ/a₀`, apply its stoichiometry, advance the
  clock, repeat.
- **Related methods** (discussed in THEORY): Gibson–Bruck next-reaction method
  (an event-queue optimization), tau-leaping (fire many events per step for
  speed at high copy numbers), the chemical Langevin equation (a diffusion
  approximation), and the reaction-diffusion master equation (RDME) for spatial
  stochastic simulation.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/stochastic-gillespie-biochemical-simulation.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/stochastic-gillespie-biochemical-simulation.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\stochastic-gillespie-biochemical-simulation.sln /p:Configuration=Release /p:Platform=x64
```

This project links only the CUDA runtime (`cudart`); no extra CUDA library is
required (the RNG is a shared, reproducible splitmix64 — see the note under
*CUDA pattern used here*).

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/gene_network.txt`, prints the
result, shows the GPU-vs-CPU agreement check, and prints a timing line to stderr.

## Data

- **Sample (committed):** `data/sample/gene_network.txt` — a tiny, **synthetic**
  model specification (rate constants + run settings) so the demo runs with zero
  downloads. It defines a birth-death gene-expression model whose answer is known
  in closed form.
- **Regenerate / resize:** `python scripts/make_synthetic.py --n-traj 1024`.
- **Real models:** `scripts/download_data.ps1` / `.sh` print instructions for
  fetching curated stochastic models from the BioModels Database (SBML).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: BioModels Database — curated stochastic SBML models
(https://www.ebi.ac.uk/biomodels); NIST Chemical Kinetics Database
(https://kinetics.nist.gov); single-molecule tracking datasets on DANDI
(https://dandiarchive.org); smFISH gene-expression data (GEO,
https://www.ncbi.nlm.nih.gov/geo/).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt). The
program simulates the ensemble on both the **GPU** (`src/kernels.cu`) and a
**CPU reference** (`src/reference_cpu.cpp`) and checks they agree:

- **Integer quantities** (each trajectory's final molecule count and number of
  events) match **exactly** — because both sides run the identical RNG stream and
  identical reaction-selection logic.
- The **time-averaged count** (a floating-point sum) matches to `~1e-15`; the
  tiny residual is host-vs-device fused-multiply-add rounding, documented in
  THEORY §Numerics. The verification tolerance (`1e-9`) is far above that residual
  and far below anything that would change the printed 4-decimal values.

As a **science check**, the ensemble mean of the time-averaged mRNA count is
printed next to the analytic stationary mean `k_prod/k_deg = 20`; with 256 short
trajectories it recovers ≈19.2 (finite-sample Monte-Carlo error), and the error
shrinks as you add trajectories.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads the config, runs CPU + GPU, verifies, reports.
2. [`src/ssa.h`](src/ssa.h) — the shared `__host__ __device__` SSA core: the RNG,
   the mass-action propensity, and the event-by-event trajectory loop. **This is
   the heart of the project.**
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the one-thread-per-trajectory idea.
4. [`src/kernels.cu`](src/kernels.cu) — the kernel and host wrapper.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the config loader, network builder, and trusted serial baseline.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, CUDA-event timer, host I/O helpers.

## Prior art & further reading

- **GillesPy2** (https://github.com/GillesPy2/GillesPy2) — Python SSA +
  tau-leaping + CLE; study its clean separation of *model* from *solver*.
- **StochPy** (https://github.com/SystemsBioinformatics/stochpy) — Python
  stochastic simulation; a readable reference for the direct and next-reaction
  methods.
- **cuTauLeaping** and CUDA-samples SSA literature — the GPU tau-leaping approach
  (one trajectory per thread, cuRAND per stream) this project mirrors in
  simplified, reproducible form.
- **MOOSE** (https://github.com/BhallaLab/moose-core) — compartmental stochastic
  kinetic simulation for neuroscience-scale models.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**One CUDA thread per trajectory**, with an independent RNG stream per thread and
the full SSA time loop in registers/local memory. There are **no atomics and no
shared memory** — trajectories are fully independent by design.

> **Note on cuRAND.** The catalog lists cuRAND (one stream per thread) and Thrust
> (propensity prefix-sum). Production GPU-SSA codes do use cuRAND for speed. This
> teaching version deliberately uses a **shared splitmix64 counter-based RNG**
> instead, so the CPU and GPU draw the *same* random numbers and the demo is
> **bit-for-bit reproducible and exactly verifiable**. Thrust's prefix-sum only
> pays off when a network has many reactions; with two reactions a linear scan is
> both simpler and faster. THEORY §"Where this sits in the real world" explains
> the cuRAND/Thrust swap and what it costs you in reproducibility.

## Exercises

1. **Grow the ensemble.** Regenerate with `--n-traj 4096` and watch the recovered
   mean converge to the analytic `k_prod/k_deg`; plot error vs. `1/√N`.
2. **Add a second species.** Extend the network to gene expression
   `DNA → mRNA → protein` (`SSA_MAX_SPECIES` is already 4). Add transcription,
   translation, and two degradation reactions in `build_gene_network`, and report
   the protein Fano factor (variance/mean).
3. **Switch to cuRAND.** Replace the shared RNG in the kernel with a per-thread
   `curandStatePhilox4_32_10_t`. You lose exact CPU==GPU verification — verify
   instead by comparing the *distributions* (mean, variance) to the analytic
   Poisson. Discuss the reproducibility trade-off.
4. **Implement tau-leaping.** Add an approximate solver that fires
   `Poisson(aⱼ·τ)` events of each reaction per fixed `τ`. Compare its speed and
   its bias against the exact SSA at high copy numbers.
5. **Add a bimolecular reaction.** Introduce dimerization `M + M → D` (order 2,
   `propensity()` already handles the `n(n-1)/2` homodimer count) and study how
   the noise changes.

## Limitations & honesty

- **Simplified model.** The committed sample is a single-species birth-death
  process — the simplest network with an analytic answer, chosen so the science
  check is exact. Real regulatory networks have many coupled species and
  nonlinear (bimolecular) reactions; the general machinery here supports up to
  `SSA_MAX_SPECIES=4` / `SSA_MAX_REACTIONS=6`, easily raised.
- **Synthetic data.** `data/sample/gene_network.txt` is synthetic and labeled as
  such; it is a model *specification*, not measured data.
- **Not cuRAND.** For didactic reproducibility we use a shared RNG, not cuRAND;
  see the note above. This is a teaching choice, not a performance recommendation.
- **Divergence.** Because different trajectories fire different numbers of events,
  warps stall on their slowest lane — the inherent cost of an *exact* method.
  Tau-leaping (Exercise 4) is the standard load-balancing remedy.
- **Not for clinical use.** Educational study material only.
