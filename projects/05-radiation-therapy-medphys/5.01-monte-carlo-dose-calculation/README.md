# 5.01 — Monte Carlo Dose Calculation (simplified slab)

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Radiation%20Therapy%20%26%20Medical%20Physics-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 5: Radiation Therapy & Medical Physics · Catalog ID `5.01`
>
> _Educational only — not for clinical use (see CLAUDE.md §8). This is a deliberately reduced-scope
> teaching model — see "Limitations"._

## Summary

Estimate how radiation **dose** is deposited with depth by tracking many photon
**histories** stochastically through a slab. Each history is independent, so each
GPU thread tracks one photon, samples its interactions with a per-thread random
number generator, and **atomically** adds its energy deposits to a shared
depth-dose tally. This is the fourth distinct GPU pattern in the flagships:
**massively parallel stochastic simulation with atomic scoring**.

## What this computes & why the GPU helps

Monte Carlo transport samples each particle's path and interactions from
probability distributions; averaging over millions of histories gives the dose.
Clinical accuracy needs ~10⁹–10¹⁰ histories — hours on a CPU. Histories are
independent, so GPUs map one thread per particle and reach ~100× speed-ups
(DPM-GPU, gDPM, MC-GPU). The key GPU challenges, both visible here, are
**execution divergence** (different particles take different branches) and
**scoring contention** (many threads deposit into the same bins).

**The parallel bottleneck** is the per-history transport loop; we run one thread
per history (grid-stride) and tally with `atomicAdd`.

## The algorithm in brief

Per photon: repeatedly sample a free-path step `s = -ln(ξ)/μ`, advance depth, and
at each interaction either **absorb** (deposit all remaining energy) or
**forward-scatter** (deposit a packet and continue). Tally **integer** energy
quanta per depth bin.

See [THEORY.md](THEORY.md) for the physics, the RNG choice, and the full real-world model.

## Build

Requires **Visual Studio 2026** (v145) + **CUDA 13.3** ([docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/monte-carlo-dose-calculation.sln`.
2. **`Release|x64`** → **Build** → `build/x64/Release/monte-carlo-dose-calculation.exe`.

CLI: `msbuild build\monte-carlo-dose-calculation.sln /p:Configuration=Release /p:Platform=x64`

## Run the demo

```powershell
./demo/run_demo.ps1
```

Simulates the committed parameter set on CPU + GPU and verifies the dose tallies match exactly.

## Data

- **Sample (committed):** `data/sample/mc_params.txt` — the slab + run parameters.
- **Realistic physics:** EGSnrc / GATE / MC-GPU (real cross sections, CT geometry) —
  see `scripts/download_data.ps1` and [data/README.md](data/README.md).
- More histories: `python scripts/make_synthetic.py --photons 4000000`.

## Expected output

`demo/expected_output.txt` holds the deterministic depth-dose histogram. The GPU
(`src/kernels.cu`) and CPU (`src/reference_cpu.cpp`) run the **identical
histories** (shared RNG in `src/mc_physics.h`) and tally **integer** quanta, so
their results are **bit-identical** (`bin mismatches = 0`) — atomic integer adds
commute, unlike float dose.

## Code tour

1. [`src/main.cu`](src/main.cu) — load params, run CPU + GPU MC, verify, print histogram.
2. [`src/mc_physics.h`](src/mc_physics.h) — **the shared RNG + photon transport** (host + device).
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU MC interface (per-thread RNG, atomic scoring).
4. [`src/kernels.cu`](src/kernels.cu) — the grid-stride history kernel + host wrapper.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the serial reference (same histories, plain add).

## Prior art & further reading

- **EGSnrc** (<https://github.com/nrc-cnrc/EGSnrc>) — reference CPU photon/electron MC.
- **GATE 10** (<https://github.com/OpenGATE/opengate>) — Geant4-based clinical MC.
- **MC-GPU** (<https://github.com/DIDSR/MCGPU>) — open CUDA photon MC dose.
- **FRED** (<https://www.fredonline.eu/>) — GPU MC for proton/ion therapy.

Study these for real physics; this project reimplements only the *pattern* didactically (CLAUDE.md §2).

## CUDA pattern used here

Per-thread RNG (reproducible streams) · independent histories (grid-stride) ·
`atomicAdd` scoring into shared bins · **integer** energy quanta for exact,
deterministic, CPU-matching tallies · branch divergence as the headline MC
challenge. (Production uses cuRAND + float dose with statistical verification.)

## Exercises

1. **Use cuRAND.** Swap the shared RNG for `curandStatePhilox4_32_10_t`. Now CPU
   and GPU diverge — verify *statistically* (within a few % per well-sampled bin)
   instead of exactly. Discuss the trade-off.
2. **Add a buildup region.** Model forward-transported secondary electrons so the
   dose rises before it falls (real depth-dose curves have a `d_max`).
3. **Layered slab.** Make `μ` depth-dependent (bone/tissue/lung layers) and watch
   the dose change at interfaces.
4. **Variance reduction.** Add Russian roulette / splitting and measure the
   variance per unit time.
5. **Divergence study.** Sort histories by remaining energy between steps and
   measure the effect on warp efficiency.

## Limitations & honesty

- **Reduced-scope teaching model:** 1-D, single material, integer energy quanta,
  one absorb/forward-scatter branch. **No real cross sections, no Compton/
  Klein-Nishina angular sampling, no electron transport, no CT geometry.**
- Because of the absorption model there is **no buildup region** (peak is at the
  surface); real photon depth-dose peaks at `d_max`. See THEORY and Exercise 2.
- We use a **shared deterministic RNG** (not cuRAND) specifically so CPU and GPU
  histories are identical for exact verification. **Not a dose engine.**
