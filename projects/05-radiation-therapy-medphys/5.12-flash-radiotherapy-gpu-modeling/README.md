# 5.12 — FLASH Radiotherapy GPU Modeling

![difficulty](https://img.shields.io/badge/difficulty-Advanced-blue) ![maturity](https://img.shields.io/badge/maturity-Frontier%2FTheoretical-informational) ![domain](https://img.shields.io/badge/domain-Radiation%20Therapy%20%26%20Medical%20Physics-lightgrey)

> **🔴 Advanced · Frontier/Theoretical** — Domain 5: Radiation Therapy & Medical Physics · Catalog ID `5.12`
>
> _Educational only — not for clinical use (see CLAUDE.md §8). This is a
> deliberately **reduced-scope teaching model** of the FLASH effect; the
> research-grade version is described in [THEORY.md](THEORY.md)._

## Summary

**FLASH radiotherapy (FLASH-RT)** delivers a whole treatment dose in a few
millisecond pulses at *ultra-high dose rate* (UHDR, > 40 Gy/s) and — in animal
studies — spares normal tissue while still controlling the tumour. This project
models one leading explanation, **transient oxygen depletion**, as a small
coupled radiation-chemistry ODE solved **per tissue voxel**. We sweep a range of
oxygen tensions (pO₂) and, for each, deliver the *same* 10 Gy either
conventionally (slowly) or in a FLASH pulse train (all at once), then compare the
resulting oxygen-fixed damage. The demo reproduces the qualitative FLASH
signature — normal-tissue sparing that is largest at low oxygenation — entirely
from the physics, with nothing about "FLASH" hard-coded. Because each voxel is an
independent ODE solve, this is a textbook **ensemble-integration** GPU workload:
one thread per voxel.

## What this computes & why the GPU helps

Modelling the FLASH effect in full requires coupled radiation-chemistry
simulation: GPU Monte-Carlo particle transport for local dose, GPU track-
structure to seed the initial radical species (OH•, e⁻ₐq, H₂O₂ …), and GPU
diffusion-reaction kinetics for oxygen depletion and radical recombination
(the MPEXS2.1-DNA code does exactly this under UHDR). We keep the *radiobiology*
— coupled radical/oxygen kinetics scored through an **oxygen-enhancement-ratio
(OER)** damage model — and reduce the transport/track-structure to a lumped
radical-yield term, so the project stays a clean, buildable teaching artifact.

**The parallel bottleneck:** the same stiff-ish chemistry ODE must be integrated
for *millions* of spatial positions × delivery conditions. Each integration is
sequential in time but **independent** of the others, so it maps perfectly onto
the GPU: **one thread integrates one voxel's full pulse-train trajectory** in
registers and writes a single result. This is the same ensemble-ODE pattern used
by the epidemiology (`9.02`) and PBPK (`13.02`) flagships.

## The algorithm in brief

- **Per-voxel state:** radical concentration `R` and oxygen `O2`, plus two
  running sums that form a *radical-weighted average oxygen* (the O₂ the DNA
  "saw" while damage was being fixed).
- **Chemistry ODE:** radical–radical recombination (`-2 k_rr R²`), radical–oxygen
  consumption (`-k_ro R·O2`, which also depletes O₂), and vascular O₂ resupply
  (`+k_diff (O2_supply − O2)`), integrated with classical **RK4**.
- **Delivery:** a train of `n_pulses` radical deposits. **Conventional** = long
  inter-pulse gaps (O₂ refills between pulses); **FLASH/UHDR** = negligible gaps
  (radicals stack and deplete O₂ before it can recover).
- **Scoring:** damage `= dose × OER(effective O2)` with the classic
  Alper–Howard–Flanders OER curve. The FLASH depletion lowers the effective O₂,
  lowers the OER, and so lowers the damage — the modelled **sparing**.
- **GPU mapping:** one thread per ensemble member (`pO₂ × {conv, FLASH}`); the
  RK4 core is shared `__host__ __device__` code, so CPU and GPU agree to ~1e-14.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/flash-radiotherapy-gpu-modeling.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/flash-radiotherapy-gpu-modeling.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\flash-radiotherapy-gpu-modeling.sln /p:Configuration=Release /p:Platform=x64
```

Only the CUDA runtime is linked — no cuBLAS/cuFFT/cuRAND — because the model is a
deterministic ODE (no random sampling), which also keeps the result bit-stable.

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/flash_ensemble.txt`, prints the
FLASH-vs-conventional table, shows the GPU-vs-CPU agreement check, and prints a
timing line to stderr.

## Data

- **Sample (committed):** `data/sample/flash_ensemble.txt` — a tiny synthetic
  **ensemble configuration** (a pO₂ sweep + beam/timing parameters). The physics
  is in the code; this file only chooses which sweep to run.
- **Regenerate/resize:** `python scripts/make_synthetic.py [--n-po2 N]`.
- **Full/real data:** `scripts/download_data.ps1` / `.sh` (they only print
  pointers — the real FLASH-RT data is credentialed/not redistributable).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: FLASH-RT experimental dosimetry from CERN/CLEAR,
UCLouvain, Stanford FLASH programs (verify access); AAPM FLASH-RT working-group
benchmark datasets (verify URL); published tumour oxygen-tension measurements;
Geant4-DNA radiolysis validation datasets.

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt): a
table of `conv_damage`, `flash_damage`, FLASH `min O2`, and the **sparing factor**
(`conv/flash`, > 1 = FLASH spared tissue) across 8 oxygen levels, ending in
`RESULT: PASS`. The program computes every member on both the **GPU**
(`src/kernels.cu`) and a **CPU reference** (`src/reference_cpu.cpp`) and asserts
they agree within `1e-9`; because both run the identical double-precision RK4
core, the real worst-case difference is ~`1e-14`.

## Code tour

Read in this order:

1. [`src/flash.h`](src/flash.h) — the shared `__host__ __device__` physics core:
   the chemistry ODE, RK4, OER, and `integrate_voxel` (start here).
2. [`src/main.cu`](src/main.cu) — loads the config, runs CPU + GPU, verifies,
   prints the deterministic table.
3. [`src/reference_cpu.h`](src/reference_cpu.h) / [`.cpp`](src/reference_cpu.cpp)
   — the ensemble config, the `(index → voxel job)` mapping, the serial baseline.
4. [`src/kernels.cuh`](src/kernels.cuh) / [`kernels.cu`](src/kernels.cu) — the GPU
   interface and the one-thread-per-voxel kernel + host wrapper.
5. [`src/util/`](src/util/) — shared `CUDA_CHECK`, CUDA-event timer, I/O helpers.

## Prior art & further reading

- **MPEXS2.1-DNA** — GPU water radiolysis under UHDR (the closest research-grade
  analogue; see the paper for the full radiolysis network this project abstracts).
- **GATE 10** (<https://github.com/OpenGATE/opengate>) — FLASH macro-dose Monte
  Carlo transport.
- **TOPAS** (<https://github.com/OpenTOPAS/OpenTOPAS>) — FLASH dosimetry extensions.
- **Geant4-DNA** (<https://github.com/Geant4/geant4>, <https://geant4-dna.org>) —
  the gold-standard track-structure micro-kinetics for FLASH-effect modelling.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Ensemble ODE integration** (PATTERNS.md §1: *"the same ODE for many parameter
sets"*). One GPU thread integrates one voxel's entire pulse-train trajectory with
RK4 in registers; the per-element physics lives in one `__host__ __device__`
header shared with the CPU reference (the HD-macro idiom, PATTERNS.md §2), so the
two produce byte-for-byte matching numbers. The catalog envisions a richer kernel
(per-voxel Gillespie SSA with cuRAND, shared-memory species arrays, CUDA streams
pipelining transport and chemistry, atomic species updates); this teaching
version keeps the *ensemble-parallelism* lesson and a **deterministic** ODE so
the demo's stdout is reproducible — the stochastic SSA variant is left as an
exercise.

## Exercises

1. **Finer sweep:** run `python scripts/make_synthetic.py --n-po2 64` and plot the
   sparing factor vs pO₂. Where is the sparing largest, and why (hint: the *slope*
   of the OER curve)?
2. **Pulse structure:** vary `n_pulses` (keeping total dose fixed). At what pulse
   count does the FLASH advantage disappear, and how does that relate to the O₂
   recovery time `1/k_diff`?
3. **Stochastic chemistry:** replace the deterministic ODE with a per-voxel
   **Gillespie SSA** (cuRAND for channel selection). Keep determinism by
   accumulating integer species counts (PATTERNS.md §3) and averaging many
   trajectories per voxel.
4. **Second precision:** run the integrator in FP32 and measure how large a
   verification tolerance you now need — a concrete lesson in FMA/rounding drift
   over thousands of steps.
5. **True spatial diffusion:** promote the single-voxel resupply term to a 3-D
   Laplacian stencil (à la flagship `14.02`) so oxygen diffuses *between* voxels,
   and watch the sparing map develop spatial structure.

## Limitations & honesty

- **Reduced scope.** This is a lumped 2-species ODE, not track-structure Monte
  Carlo. Particle transport and the initial radical spectrum are collapsed into a
  single `g_rad` yield; the dozens of real radiolysis species and their spatial
  diffusion are not modelled.
- **Illustrative parameters.** The rate constants and yields (`src/flash.h`
  `default_rates`) are chosen so the demo is *interpretable*, **not fitted** to
  any measurement. Absolute damage is in arbitrary "Gy-equivalent" units; only the
  conventional-vs-FLASH **ratio** carries meaning, and even that is qualitative.
- **Synthetic data, labelled as such.** The committed sample is synthetic
  (`data/README.md`). No real patient or dosimetry data is used.
- **Not clinical.** Nothing here may inform diagnosis, treatment, or dose
  prescription. The FLASH effect itself remains an active research question; this
  project teaches the *computational pattern*, not a validated biological model.
