# 5.13 — BNCT Dose Calculation & Optimization

![difficulty](https://img.shields.io/badge/difficulty-Advanced-blue) ![maturity](https://img.shields.io/badge/maturity-Frontier%2FTheoretical-informational) ![domain](https://img.shields.io/badge/domain-Radiation%20Therapy%20%26%20Medical%20Physics-lightgrey)

> **🔴 Advanced · Frontier/Theoretical** — Domain 5: Radiation Therapy & Medical Physics · Catalog ID `5.13`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

This project is a **reduced-scope teaching Monte Carlo** for **Boron Neutron
Capture Therapy (BNCT)** dose calculation. It fires simulated neutrons into a
1-D tissue slab, tracks each one as it slows down and is eventually captured, and
tallies the four physically distinct BNCT dose components — **boron**
(¹⁰B(n,α)⁷Li, the therapeutic reaction), **nitrogen**, **hydrogen-capture
gamma**, and **fast-neutron recoil** — as a function of depth. It then reports
the **CBE/RBE-weighted biological dose** (Gy-Eq). Every neutron history is
independent, so the simulation maps naturally onto the GPU: **one thread per
history**, with results scored via `atomicAdd` into a shared dose tally. Because
energy is accumulated in **integer keV quanta**, the GPU result matches the CPU
reference **exactly** — a clean, zero-tolerance verification that teaches the
"integer atomics = deterministic reduction" lesson. It is educational only, not a
clinical dose engine (see [THEORY.md](THEORY.md) §7).

## What this computes & why the GPU helps

Boron Neutron Capture Therapy (BNCT) delivers therapeutic dose by targeting tumor cells loaded with ¹⁰B, which captures thermal neutrons to release high-LET alpha particles and lithium recoils. Dose calculation involves: (1) neutron transport (diffusion or discrete ordinates / Monte Carlo) to compute thermal neutron flux maps, (2) boron dose from ¹⁰B(n,α)⁷Li reaction rates, (3) high-LET photon dose, and (4) fast neutron dose — each requiring separate cross-section libraries and requiring GPU-parallel transport. The compound biological effectiveness (CBE) factor and boron uptake heterogeneity add biological modeling complexity. Treatment planning must jointly optimize beam direction and boron carrier dosing.

**The parallel bottleneck:** simulating the **neutron histories**. A useful dose
estimate needs 10⁶–10¹⁰ independent histories, and each is an independent
random walk (sample a free path, decide scatter vs. capture, deposit energy,
repeat). That is the classic *embarrassingly parallel* Monte-Carlo workload:
we give **each history its own GPU thread** (a grid-stride loop over millions of
them) and score deposits with `atomicAdd`. The transport per history is cheap;
the win comes from running a huge *number* of them concurrently.

## The algorithm in brief

- **Monte Carlo neutron transport** — sample each neutron's exponential free
  path `s = -ln(ξ)/Σ_tot`, advance it, and decide the interaction by
  cross-section shares (this teaching version uses **two energy groups**, fast +
  thermal, in a 1-D slab).
- **Thermalization** — fast neutrons elastically scatter (mostly off hydrogen)
  until they slow to thermal energy; a fast scatter deposits a recoil-proton
  dose quantum.
- **Capture & dose components** — a thermal capture is assigned to ¹⁰B, ¹⁴N, or
  ¹H in proportion to each nuclide's macroscopic capture cross section `Σ_a`,
  depositing that reaction's energy into the matching dose component.
- **CBE/RBE-weighted biological dose** — the four physical component doses are
  combined with their biological weights, `D_bio = Σ_c w_c·D_c` (Gy-Eq).
- **GPU pattern** — one thread per neutron history (grid-stride), integer-quanta
  `atomicAdd` scoring for a deterministic, exactly-verifiable tally.

The catalog also lists discrete-ordinates (Sₙ) transport, full ENDF/B-VIII
cross-section libraries, and joint beam+boron optimization; those belong to the
production version described in [THEORY.md](THEORY.md) §7.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/bnct-dose-calculation-optimization.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/bnct-dose-calculation-optimization.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\bnct-dose-calculation-optimization.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/`, prints the result, shows the
GPU-vs-CPU agreement check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/bnct_params.txt` — a tiny, **synthetic**,
  offline parameter file (15 numbers) so the demo runs with zero downloads.
  Regenerate/scale with `python scripts/make_synthetic.py`.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print pointers to real
  BNCT references (OpenMC, GATE, ENDF/B-VIII, IAEA benchmarks); none is needed
  for the demo, and gated data is never bypassed.
- **Provenance, field meanings & license:** see [data/README.md](data/README.md).

Catalog dataset notes: IAEA BNCT benchmark cases (verify URL at iaea.org); BNCT clinical trial CT data from Finnish accelerator BNCT program; OpenMC validation datasets (https://github.com/openmc-dev/openmc/tree/develop/tests); NIST neutron cross-section data (verify URL).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt): the
four component totals with their dose shares, the CBE/RBE-weighted biological
dose, and the boron depth-dose curve (which rises to a sub-surface peak then
falls — the thermal-neutron build-up). The program computes the tally on both the
**GPU** (`src/kernels.cu`) and a **CPU reference** (`src/reference_cpu.cpp`) and
asserts they agree **exactly** — because the dose is accumulated in integer keV
quanta, the tolerance is **zero mismatches**, not an approximate float bound.
That exact agreement is the correctness guarantee.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads data, runs CPU + GPU, verifies, reports.
2. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the thread-mapping idea.
3. [`src/kernels.cu`](src/kernels.cu) — the kernel(s) and host wrapper.
4. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline.
5. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

OpenMC (https://github.com/openmc-dev/openmc) — open-source GPU-capable neutron MC (OpenMP/GPU via OpenMP target offload); GATE 10 (https://github.com/OpenGATE/opengate) — neutron transport for BNCT; COMPASS BNCT MC (verified in Nature Scientific Reports, https://pmc.ncbi.nlm.nih.gov/articles/PMC10366114/); OpenMC MeVisLab BNCT pipeline (https://www.hplpb.com.cn/en/article/doi/10.11884/HPLPB202537.250246).

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Per-thread Monte-Carlo histories + atomic scoring** (docs/PATTERNS.md §1, as in
flagship 5.01): one GPU thread per neutron history via a grid-stride loop, a
private reproducible RNG per thread (shared `__host__ __device__` header so
CPU==GPU), and `atomicAdd` into a small global dose tally. The catalog also
mentions cross-section tables in texture memory, a 3-D boron map, cuBLAS for
multi-group flux equations, and material-sorted particle batches to fight warp
divergence — those are production optimizations described in
[THEORY.md](THEORY.md) §4/§7 and left as exercises.

## Exercises

1. **Tumor selectivity.** Make `Σ_a,B` **depth-dependent** — high only in a
   "tumor" band of bins, near-zero elsewhere — and show the boron dose spikes
   inside the tumor while the background components stay flat. This is the core
   BNCT idea. (Requires passing a per-bin boron array instead of a scalar.)
2. **Convergence.** Sweep `n_histories` (1e4 → 1e7) and plot the boron
   depth-dose's statistical noise shrinking as ~1/√N. Note where the GPU starts
   to clearly beat the CPU (small N is launch-bound; see `docs/PATTERNS.md §7`).
3. **Fight divergence.** Sort/bin histories by whether they thermalize before
   scoring, or process fast and thermal phases in separate kernels, and measure
   the effect on kernel time (warp-divergence mitigation from THEORY §4).
4. **Reduce atomic contention.** Give each block a shared-memory sub-tally and
   flush once per block — compare timing for a much larger `n_bins`.
5. **Better physics.** Add a third energy group (epithermal), or replace the flat
   two-group cross sections with a small tabulated Σ(E) read into constant/texture
   memory — a step toward the multi-group libraries real codes use.

## Limitations & honesty

- **Reduced-scope teaching model, not a dose engine.** It is **1-D** (a slab, not
  a voxelized patient), uses **two flat energy groups** (not continuous-energy
  ENDF/B cross sections), and deposits reaction energy **locally** (kerma
  approximation — it does not transport the α/⁷Li/proton secondaries).
- **All data is synthetic.** The cross sections in `data/sample/bnct_params.txt`
  are order-of-magnitude realistic but are **not** a validated library; the
  `gray_per_keV` scale is arbitrary. The reported Gy / Gy-Eq numbers are a
  teaching artifact, **not** a real patient dose.
- **No clinical claim.** Nothing here may inform diagnosis or treatment
  (CLAUDE.md §8). The CBE/RBE weights are representative literature values, not a
  patient- or drug-specific calibration.
- **Timing is illustrative**, not a benchmark (CLAUDE.md §12). Production BNCT
  differs on every axis described in [THEORY.md](THEORY.md) §7.
