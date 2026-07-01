# 5.10 — Secondary Cancer Risk & Stray-Dose Monte Carlo

![difficulty](https://img.shields.io/badge/difficulty-Advanced-blue) ![maturity](https://img.shields.io/badge/maturity-Frontier%2FTheoretical-informational) ![domain](https://img.shields.io/badge/domain-Radiation%20Therapy%20%26%20Medical%20Physics-lightgrey)

> **🔴 Advanced · Frontier/Theoretical** — Domain 5: Radiation Therapy & Medical Physics · Catalog ID `5.10`
>
> _Educational only — not for clinical use (see CLAUDE.md §8). This is a
> **reduced-scope teaching version** of a research-grade problem (CLAUDE.md §13);
> the full version is described in [THEORY.md](THEORY.md)._

## Summary

Radiotherapy aims a high dose at a tumour, but radiation does not stop at the field
edge: **stray radiation** (scatter inside the body, leakage through the machine
head, and — in proton/high-energy therapy — secondary neutrons) sprinkles a tiny
dose over the whole body, three to five orders of magnitude below the target dose.
Because that dose acts on large volumes of healthy tissue over a patient's lifetime,
it drives **secondary cancer risk**. This project is a GPU Monte Carlo that
transports primary photons through a reduced 1-D "organ-stack" phantom, scores the
out-of-field stray dose in each organ, and convolves it with a BEIR-VII-style
lifetime-risk model to estimate secondary-cancer risk. Its didactic heart is
**variance reduction** — survival biasing, Russian roulette, and forced detection —
the techniques that make a rare-signal calculation tractable at all.

## What this computes & why the GPU helps

Radiotherapy delivers dose not only to the target but also to distant organs via
stray radiation (leakage, scatter, neutrons from proton therapy nuclear
interactions), creating secondary cancer risk. Stray-dose is ~3–4 orders of
magnitude lower than target dose, requiring 10¹¹–10¹²+ particle histories per
calculation for statistical precision — intractable even on GPU without variance
reduction (splitting, forced detection, geometry importance).

**The parallel bottleneck:** the calculation is a sum over an enormous number of
**independent particle histories**. Each history is a short random walk (sample a
step, interact, scatter/absorb, roulette). Independence is what makes it a perfect
GPU job: one thread per history, grid-strided over millions of them, each with its
own RNG stream. The scarcity of the stray signal is handled *algorithmically*
(variance reduction), not by brute force — so we teach both the parallel mapping
**and** why naive analog MC would need 10¹¹⁺ histories.

## The algorithm in brief

- **Survival biasing:** photons are never simply killed on absorption; they carry a
  fractional statistical **weight** that shrinks at each interaction and keep going.
- **Russian roulette:** once the weight drops below a floor, randomly terminate or
  boost it — an unbiased way to stop tracking negligible particles.
- **Forced detection (next-event scoring):** at every scatter site, *deterministically*
  add the expected stray contribution to each downstream out-of-field organ instead
  of waiting for a rare lateral-scatter event — the big variance win.
- **Leakage + neutron channels:** a uniform machine-leakage bath plus a
  distance-weighted secondary-neutron **surrogate** (the full hadronic INCL/BERT
  cascade is out of scope — see THEORY.md).
- **BEIR-VII risk convolution:** organ dose × per-organ risk coefficient, summed over
  out-of-field organs, under the Linear No-Threshold assumption.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/secondary-cancer-risk-stray-dose-monte-carlo.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/secondary-cancer-risk-stray-dose-monte-carlo.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\secondary-cancer-risk-stray-dose-monte-carlo.sln /p:Configuration=Release /p:Platform=x64
```

This project links only the CUDA runtime (`cudart_static.lib`); it uses no extra
CUDA library (the RNG is a hand-rolled splitmix64 shared by CPU and GPU for exact
verification — see THEORY.md "Numerical considerations").

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/phantom.txt`, prints the per-organ
stray dose and secondary-cancer risk, shows the GPU-vs-CPU agreement check, and
prints a timing line.

## Data

- **Sample (committed):** `data/sample/phantom.txt` — a tiny, **synthetic** 1-D
  organ-stack phantom so the demo runs offline with zero downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` (prints instructions;
  never bypasses registration).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: ICRP 110 voxel phantoms (adult male/female,
<https://www.icrp.org/publication.asp?id=ICRP%20Publication%20110>); NIST photon
cross-section databases (<https://www.nist.gov/pml/xcom-photon-cross-sections>);
secondary dose measurements from literature; TCIA proton therapy planning CTs.

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt). The
program simulates the **identical** histories on the **GPU** (`src/kernels.cu`) and
a **CPU reference** (`src/reference_cpu.cpp`) and asserts the per-organ dose tallies
are **bit-identical** — deposits are fixed-point integers, so the GPU's atomic sums
commute and match the CPU exactly (tolerance = 0). That exact agreement is the
correctness guarantee; `RESULT: PASS` reports it.

## Code tour

Read in this order:

1. [`src/stray_physics.h`](src/stray_physics.h) — the shared host+device RNG and
   the whole per-history transport (survival biasing, forced detection, roulette).
   **Start here** — this is the science.
2. [`src/risk_model.h`](src/risk_model.h) — the BEIR-VII dose → lifetime-risk map.
3. [`src/main.cu`](src/main.cu) — loads the phantom, runs CPU + GPU, verifies, reports.
4. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the thread-mapping idea.
5. [`src/kernels.cu`](src/kernels.cu) — the kernel (grid-stride histories + atomic
   fixed-point scoring) and its host wrapper.
6. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the loader + trusted serial baseline.
7. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

- **TOPAS** (<https://github.com/OpenTOPAS/OpenTOPAS>) — Geant4-based, full hadronic
  transport and stray-dose extensions. Study how a production code layers geometry,
  physics lists, and scoring; we hand-roll a reduced 1-D analogue.
- **GATE 10** (<https://github.com/OpenGATE/opengate>) — neutron transport and
  out-of-field dose scoring; a reference for what "out-of-field scoring" means.
- **EGSnrc** (<https://github.com/nrc-cnrc/EGSnrc>) — the classic photon/electron MC
  with mature variance reduction; the source for the survival-biasing / roulette /
  forced-detection idioms we reimplement.
- **PHITS** (<https://phits.jaea.go.jp/>) — hadronic + neutron transport used in
  radiation protection; context for the neutron channel we only surrogate here.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Per-thread RNG + atomic scoring (PATTERNS.md §1, exemplar flagship
`5.01`): one thread per primary history, grid-strided over millions; each thread
seeds a reproducible splitmix64 stream from its history index and runs the shared
`__host__ __device__` transport; deposits are accumulated with `atomicAdd` into a
small per-organ global tally. **Fixed-point integer** deposits make the atomics
order-independent, so the GPU result is deterministic and equals the CPU reference
exactly. Variance reduction (survival biasing, roulette, forced detection) is done
per-thread inside the history — no particle-stack splitting is needed for this
teaching scope (the full particle-forking version is discussed in THEORY.md).

## Exercises

1. **Vary the field size.** Change `field_end` in `data/sample/phantom.txt` (e.g.
   organs 0–2 in-field). How does the out-of-field dose profile shift?
2. **Turn variance reduction off.** Set `sidescatter` and the leakage/neutron
   channels to 0 and raise `n_histories`; watch the distant-organ tallies get noisy.
   Estimate how many analog histories you'd need for the same precision.
3. **Add photon energy.** Replace the single constant `mu` with an energy-dependent
   attenuation (a small lookup table in constant memory) and let the RNG sample it.
4. **Move risk onto the GPU.** `organ_lar()` is already `__host__ __device__`; add a
   tiny reduction kernel that sums out-of-field LAR on the device. Verify it matches
   the host sum exactly.
5. **Confidence intervals.** Accumulate per-organ sum-of-squares (also fixed-point)
   and report a Monte-Carlo standard error alongside each dose.

## Limitations & honesty

- **Reduced scope on purpose.** This is a **1-D organ-stack** phantom with a single
  constant attenuation coefficient — not a 3-D ICRP-110 voxel phantom with
  energy-dependent NIST cross-sections. The transport is a deliberately simplified
  photon walk; there is no electron transport and no true hadronic cascade.
- **The neutron channel is a surrogate.** Secondary neutrons are modelled as an
  extra distance-weighted leakage term, not real INCL/BERT hadronic physics
  (that needs Geant4/TOPAS and is genuinely research-grade).
- **Everything is synthetic.** The phantom, the parameters, and the risk
  coefficients are illustrative teaching values, **not** clinical BEIR-VII
  coefficients. No output may inform any medical decision.
- **Timing is a teaching artifact**, not a benchmark — on the tiny committed sample
  the GPU's per-history parallelism competes with launch overhead; the GPU's edge
  grows with history count (real plans run 1e9–1e12 histories).
