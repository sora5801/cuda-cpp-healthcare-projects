# 1.24 — Umbrella Sampling / WHAM Free Energy Profiles

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Drug%20Discovery%20%26%20Molecular%20Design-lightgrey)

> **🟢 Beginner · Established** — Domain 1: Drug Discovery & Molecular Design · Catalog ID `1.24`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

A molecule that binds, unbinds, or permeates moves along a **reaction coordinate**,
and the quantity that decides how strongly it binds is the **free-energy profile**
along that coordinate — the *potential of mean force* (PMF). Plain simulation
rarely samples the top of a free-energy barrier, so the PMF there is invisible.
**Umbrella sampling** fixes this by running many independent simulations, each held
near a different point on the coordinate by a harmonic "umbrella" restraint; each
produces a histogram of where it visited. **WHAM** (the Weighted Histogram Analysis
Method) then stitches those biased histograms back into one unbiased PMF. This
project implements the whole pipeline on a **synthetic double-well** landscape: the
GPU runs one biased trajectory per window (the *ensemble* pattern), and the CPU
runs WHAM. Because we know the true landscape, we can check that WHAM actually
recovers it.

## What this computes & why the GPU helps

Umbrella sampling applies harmonic restraints along a reaction coordinate (e.g.,
ligand unbinding distance, pore radius) to force sampling at energy barriers.
Multiple windows run simultaneously — **embarrassingly parallel across windows** —
each an independent biased simulation. WHAM post-processes the window histograms
into a PMF. GPU dynamics lets each window generate far more biased trajectory per
second, so the PMF converges in cases that were previously impractical.
Applications include permeation barriers in ion channels and drug binding/unbinding
free-energy profiles.

**The parallel bottleneck:** the cost is *generating the biased trajectories* —
one long sequential time-loop per window. The windows are mutually independent, so
we map **one GPU thread to one window** and run all windows at once. (WHAM itself
is cheap `O(iters · windows · bins)` post-processing and stays on the CPU, exactly
as production pipelines do.) In this teaching demo the per-window work is tiny, so
the GPU is actually *slower* than the CPU here — an honest, instructive result; see
**Limitations**.

## The algorithm in brief

- **Harmonic bias potentials** — each window adds `w_k(x) = ½ k (x − x0_k)²`,
  tethering the system near center `x0_k` even on top of a barrier.
- **Overdamped Langevin dynamics** — the biased trajectory generator (the
  statistical-mechanics skeleton of MD, without the atoms).
- **Per-window histogramming** — count visited reaction-coordinate bins (integer
  counts → deterministic, exactly reproducible).
- **WHAM self-consistent iteration** — solve for the per-window free-energy shifts
  that make all windows' reweighted densities consistent, then `F = −kT ln p`.
- *(Mentioned in THEORY, not implemented here: MBAR, steered MD + Jarzynski,
  metadynamics.)*

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/umbrella-sampling-wham-free-energy-profiles.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/umbrella-sampling-wham-free-energy-profiles.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\umbrella-sampling-wham-free-energy-profiles.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (CMake build)
```

The demo builds if needed, runs on `data/sample/umbrella.txt`, prints the WHAM PMF
beside the analytic potential, shows the GPU-vs-CPU agreement check, and prints a
timing line on stderr.

## Data

- **Sample (committed):** `data/sample/umbrella.txt` — a tiny synthetic experiment
  configuration (double-well + window layout + dynamics settings) so the demo runs
  with zero downloads.
- **Full dataset:** there is nothing to download — `scripts/download_data.ps1` /
  `.sh` print pointers to real umbrella-sampling tools/datasets;
  `scripts/make_synthetic.py` regenerates or enlarges the sample.
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: ion-channel permeation benchmark sets; SAMPL binding
free-energy challenges (<https://github.com/samplchallenges/SAMPL>); BindingDB
(<https://www.bindingdb.org>); GROMACS umbrella-sampling tutorials
(<https://tutorials.gromacs.org>).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt). The
program computes the per-window histograms on both the **GPU** (`src/kernels.cu`)
and a **CPU reference** (`src/reference_cpu.cpp`) and asserts they are
**bit-for-bit identical** (integer counts + shared physics). It then runs WHAM and
checks the reconstructed PMF recovers the **known** double-well — the barrier comes
back at 4.00 kT and the wells at ~0 kT, within 0.30 kT over the interior of the
scan. Both checks passing prints `RESULT: PASS`.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads the experiment, runs CPU + GPU sampling,
   verifies histograms, runs WHAM, compares the PMF to the analytic `U`, reports.
2. [`src/umbrella.h`](src/umbrella.h) — **the shared `__host__ __device__` core**:
   RNG, double-well potential, biased Langevin step, per-window histogramming. This
   is what makes the CPU and GPU results identical.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the one-thread-per-
   window idea.
4. [`src/kernels.cu`](src/kernels.cu) — the kernel and host wrapper.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the serial baseline **and**
   the WHAM solver (shared by both paths).
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, CUDA-event timer, host I/O.

## Prior art & further reading

- **GROMACS `gmx wham`** (<https://github.com/gromacs/gromacs>) — the reference
  production WHAM implementation; study its window/overlap diagnostics.
- **OpenMM umbrella-sampling cookbook** (<https://github.com/openmm/openmm-cookbook>)
  — how harmonic restraints on a collective variable are set up in a real MD engine.
- **alchemlyb** (<https://github.com/alchemistry/alchemlyb>) — clean Python
  MBAR/WHAM post-processing; a good place to see MBAR (the modern successor to WHAM).
- **PLUMED** (<https://github.com/plumed/plumed2>) — collective variables + biases
  (umbrella, metadynamics) as a plugin to many MD engines.

Study these to learn the production approach; **do not copy code wholesale** —
this project reimplements the ideas didactically (CLAUDE.md §2).

## CUDA pattern used here

**Ensemble (one thread per independent simulation)** + **per-thread histogram with
no contention**. Each window is a sequential trajectory but independent of the
others, so thread `k` runs window `k`'s whole biased Langevin loop and writes into
its *own* slice of the histogram array — no atomics needed (contrast project `5.01`,
where many threads share one tally and must `atomicAdd`). Counts are integers, so
the result is deterministic and exactly matches the CPU. WHAM runs on the CPU. See
`docs/PATTERNS.md` §1–§3 and [THEORY.md](THEORY.md) for the mapping.

> The catalog's full vision ("full MD per window on GPU; MPI + NCCL to launch the
> window array; shared-memory reductions for collective-variable forces") is the
> production target — described in THEORY §"Where this sits in the real world".
> This is the **reduced-scope teaching version** (CLAUDE.md §13).

## Exercises

1. **Tighten the PMF.** Re-run with `python scripts/make_synthetic.py --n-sample
   200000` and watch the worst `|WHAM − U|` (on stderr) shrink — finite-sampling
   error falls like `1/√N`.
2. **Break the overlap.** Reduce `--n-windows` to 7 (or weaken `--k-spring` to 3).
   Neighbouring histograms stop overlapping, WHAM's barrier estimate degrades, and
   some bins go *unsampled*. This is the single most important practical lesson in
   umbrella sampling: **windows must overlap**.
3. **Change the landscape.** Make the wells asymmetric by raising the barrier
   (`--A 8`) and confirm WHAM still recovers it (you may need more sampling).
4. **Block-size sweep.** Change `THREADS_PER_BLOCK` in `kernels.cu` (32 → 256) and
   note it barely matters here — the kernel is occupancy-starved at 27 threads.
   What does that tell you about when this kernel would benefit from more windows?
5. **Add MBAR.** Implement the MBAR estimator (THEORY §"real world") and compare its
   PMF to WHAM's on the same histograms.

## Limitations & honesty

- **Synthetic landscape.** The double-well is a teaching model, not a real molecule
  or channel. Nothing here is a free-energy prediction for any real system, and
  nothing is clinical.
- **Reduced physics.** We use 1-D overdamped Langevin dynamics, not all-atom MD.
  That keeps the statistical mechanics (biased Boltzmann sampling → WHAM) exact
  while dropping the atomistic forces a real study needs.
- **The GPU is slower here, on purpose.** With only 27 windows the kernel cannot
  fill the GPU; the timing is a teaching artifact (`docs/PATTERNS.md` §7), never a
  benchmark claim. Real runs have hundreds of windows each doing thousands of force
  evaluations per step — the regime where the GPU wins.
- **WHAM, not MBAR.** We implement classic histogram WHAM; modern pipelines prefer
  MBAR (binless, lower variance). WHAM is the clearer first lesson.
- **Edges are noisier.** The outermost windows have a one-sided well, so we judge
  the PMF on the *interior* of the scan and say so (THEORY §"How we verify").
