# 1.6 — Enhanced Sampling — Metadynamics & Replica Exchange

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Drug%20Discovery%20%26%20Molecular%20Design-lightgrey)

> **🟢 Beginner · Established** — Domain 1: Drug Discovery & Molecular Design · Catalog ID `1.6`
>
> _Educational only — not for clinical use (see CLAUDE.md §8). This is a
> clearly-labeled **reduced-scope teaching version** on a synthetic 1-D model;
> production enhanced sampling is described in [THEORY.md](THEORY.md)._

## Summary

Plain molecular dynamics gets **stuck**: crossing a free-energy barrier (say, a
drug unbinding from its pocket) is a rare event that can take milliseconds of
simulated time — far beyond what MD can reach. **Well-tempered metadynamics**
escapes the trap by periodically dropping small Gaussian "hills" of bias along a
chosen reaction coordinate (a *collective variable*, CV). The bias fills up the
basin the system sits in and pushes it over the barrier — and, beautifully, the
accumulated bias **reconstructs the free-energy surface** (FES) at the same time.
This project runs an **ensemble of independent metadynamics walkers** on a
synthetic 1-D double-well landscape (one GPU thread per walker), recovers the
FES, and verifies it against the *known* analytic surface.

## What this computes & why the GPU helps

We integrate `M` independent **Langevin walkers** on the double well
`F0(s) = A(s²−1)²`, each depositing well-tempered Gaussian hills on the metadynamics
"pace". The recovered free energy is `F(s) = −(γ/(γ−1))·V_bias(s)`, where `γ` is
the bias factor. Because we *know* `F0`, we can confirm metadynamics recovers it.

**The bottleneck the GPU parallelizes:** each walker is a sequential time loop,
but the walkers are **mutually independent** — exactly the "ensemble of
trajectories" workload. The CPU runs `M` walkers one after another (`O(M·N)`
total steps); the GPU runs all `M` **at once**, so wall-clock is set by a single
walker. This is the standard pattern for multi-walker enhanced sampling and for
all uncertainty-quantification sweeps (cf. flagships 9.02, 13.02).

## The algorithm in brief

- **Langevin dynamics** with a symmetric (BAOAB-style) thermostat splitting.
- **Well-tempered metadynamics**: tempered Gaussian hill height
  `w_eff = w·exp(−V_bias(s*)/(kT(γ−1)))` → guaranteed convergence of the bias.
- **Grid bias**: store `V_bias` on a uniform grid so reads are O(1) (not O(#hills)).
- **Counter-based RNG** (SplitMix64 + Box–Muller) for deterministic, schedule-
  independent Gaussian noise shared by CPU and GPU.
- **FES recovery + reweighting**: `F(s) = −(γ/(γ−1))V_bias(s)`, min-shifted to 0.

See [THEORY.md](THEORY.md) for the science, math, GPU mapping, and numerics.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(the ratified repo standard; see [`docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md)).

1. Open `build/enhanced-sampling-metadynamics-replica-exchange.sln` in VS 2026.
2. Select **`Release|x64`** (or `Debug|x64`).
3. **Build → Build Solution**. The `.exe` lands in `build/x64/Release/`.

No manual path edits: the project uses the CUDA `.props/.targets` integration and
`$(CUDA_PATH)`. A cross-platform `CMakeLists.txt` is provided as a bonus (the VS
solution is the required deliverable). Both builds pass `--fmad=false` (see
[THEORY.md](THEORY.md) "Numerical considerations" for why).

## Run the demo

One command (builds if needed, runs on the committed sample, diffs stdout):

```powershell
# Windows (PowerShell)
./demo/run_demo.ps1
```

```bash
# Linux / macOS (CMake path)
./demo/run_demo.sh
```

Expected: `[run_demo] PASS: output matches expected_output.txt and GPU==CPU.`
See [`demo/README.md`](demo/README.md) for what the output means.

## Data

The "data" is the **run configuration** (one line:
`A kT mass friction dt steps hill_w hill_sigma deposit_every bias_factor s_lo s_hi nbins n_walkers seed s_start`),
committed at [`data/sample/metad_config.txt`](data/sample/metad_config.txt) and
**clearly synthetic** — there is no real MD trajectory, by design, so the known
analytic FES can be used for verification. Regenerate or rescale it with
[`scripts/make_synthetic.py`](scripts/make_synthetic.py)
(e.g. `--n-walkers 256 --steps 40000`). Real enhanced-sampling inputs (PLUMED-NEST,
GPCRmd) are pointed to — not bypassed — by
[`scripts/download_data.ps1`](scripts/download_data.ps1) /
[`.sh`](scripts/download_data.sh). Provenance and field meanings:
[`data/README.md`](data/README.md). _No license restrictions — fully synthetic._

## Expected output

```
1.6 -- Enhanced Sampling -- Metadynamics & Replica Exchange
well-tempered metadynamics on a 1-D double well (SYNTHETIC model)
ensemble: 64 walkers x 20000 steps; barrier A=5.00 kT, gamma=10.0, pace=50, sigma=0.10
grid: 121 bins over s in [-2.00, 2.00]; hills/walker=400
recovered FES F(s) [kT] at s = -1.0 -0.5  0.0 +0.5 +1.0:
  est :  0.0  3.0  5.0  3.0  0.4
  true:  0.0  2.8  5.0  2.8  0.0
barrier height: recovered 5.0 kT vs true 5.0 kT
RESULT: PASS (CPU & GPU recover the same FES within 0.25 kT; barrier matches analytic within 0.35 kT)
```

The recovered FES (`est`) tracks the known double well (`true`), and the **5.0 kT
barrier is recovered**. Two checks gate `PASS`: (1) the CPU and GPU recover the
same surface to within 0.25 kT over the well-sampled core; (2) the recovered
barrier matches the analytic value within 0.35 kT. **Why not bit-exact GPU==CPU?**
The Langevin trajectories are *chaotic*, so individual paths diverge across
platforms — we verify the robust **ensemble FES** instead, and print the chaotic
per-walker numbers to stderr. See [THEORY.md](THEORY.md) "How we verify".

## Code tour

Read in this order:

1. [`src/metad.h`](src/metad.h) — **the heart**: the shared `__host__ __device__`
   physics (double-well force, bias grid, well-tempered hill deposition, the
   counter-based RNG, the Langevin step, and `run_walker()`). CPU and GPU call the
   *same* code from here, which is why their results are comparable.
2. [`src/main.cu`](src/main.cu) — loads the config, runs CPU + GPU, verifies, and
   prints the deterministic report (stdout) and diagnostics (stderr).
3. [`src/kernels.cuh`](src/kernels.cuh) / [`src/kernels.cu`](src/kernels.cu) — the
   ensemble kernel (one thread per walker) and its host wrapper.
4. [`src/reference_cpu.h`](src/reference_cpu.h) /
   [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the config loader and the
   serial baseline the GPU is checked against.
5. [`src/util/`](src/util/) — shared `CUDA_CHECK`, CUDA-event timing, host I/O.

## Prior art & further reading

- **[PLUMED](https://github.com/plumed/plumed2)** — the standard plugin that
  implements CVs and bias (metadynamics, funnel MetaD, REST2…) for GROMACS, NAMD,
  OpenMM, LAMMPS. Study how it computes a bias on the host with negligible overhead.
- **[GROMACS](https://github.com/gromacs/gromacs)** — the GPU MD engine; learn how
  forces/integration run on the GPU while PLUMED supplies the bias.
- **[OpenPathSampling](https://github.com/openpathsampling/openpathsampling)** —
  transition-path sampling, a complementary rare-event approach.
- **[HTMD](https://github.com/Acellera/htmd)** — high-throughput *adaptive*
  sampling on GPU clusters (a different way to beat the rare-event problem).
- Foundational papers: Laio & Parrinello 2002 (metadynamics); Barducci, Bussi &
  Parrinello 2008 (well-tempered); Sugita & Okamoto 1999 (REMD).

## Exercises

1. **Shared-bias multi-walker MetaD.** Make all walkers deposit into *one* shared
   bias grid (atomic adds) instead of private grids. Measure how much faster the
   FES converges — this is how production multi-walker runs work.
2. **Sweep the bias factor `γ`.** Run with `--bias-factor 5, 10, 50` and watch how
   the convergence speed and FES smoothness change. Plot the recovered FES.
3. **Add a second CV.** Extend the grid to 2-D `(s1, s2)` on a coupled landscape
   and deposit 2-D Gaussians. This is where metadynamics earns its keep.
4. **Implement temperature REMD.** Run `R` replicas at different `kT`, add a
   Metropolis swap step between adjacent replicas, and recover the FES from the
   coldest replica. Compare convergence to metadynamics.
5. **Coalesce the bias reads.** The current per-walker global-memory layout is
   uncoalesced. Reorganize `d_bias` (or tile a block of walkers' grids into shared
   memory) and measure the bandwidth improvement.

## Limitations & honesty

- **Reduced scope.** This is a **1-D analytic double well**, not a real molecular
  system. There is **no force field, no atoms, no real CV** — the landscape is
  synthetic *on purpose* so the recovered FES can be checked against a known
  answer. Production enhanced sampling (PLUMED + GPU MD on 10⁴–10⁶ atoms) is
  described in [THEORY.md](THEORY.md) "Where this sits in the real world".
- **Independent walkers, not shared bias.** Real multi-walker MetaD shares one
  growing bias (needs atomics + sync). We keep walkers independent for a clean
  teaching kernel; Exercise 1 adds sharing.
- **Replica exchange is described, not coded.** The runnable code is metadynamics;
  REMD/HREX/REST2 are explained in THEORY and left as Exercise 4.
- **Chaotic trajectories are not platform-reproducible.** Only the *ensemble FES*
  is verified across CPU/GPU; per-walker endpoints are machine-local (stderr).
- **Timing is a teaching artifact**, never a benchmark claim (CLAUDE.md §12). With
  only 64 walkers the GPU is launch-bound and not faster than the CPU; its edge
  grows with the walker count.
- **Not for clinical use.** Nothing here is a diagnostic or therapeutic tool.
