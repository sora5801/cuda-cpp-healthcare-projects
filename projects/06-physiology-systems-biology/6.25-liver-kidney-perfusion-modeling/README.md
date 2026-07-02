# 6.25 — Liver & Kidney Perfusion Modeling

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Computational%20Physiology%20%26%20Systems%20Biology-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 6: Computational Physiology & Systems Biology · Catalog ID `6.25`
>
> _Educational only — not for clinical use (see CLAUDE.md §8). All data are synthetic._

## Summary

A liver lobule is thousands of tiny parallel capillaries (**sinusoids**) that clear
a drug from blood as it flows past metabolizing liver cells. This project simulates
one lobule as an **ensemble of sinusoids**: each is a 1-D convection-reaction ODE —
blood carries the drug downstream while zone-dependent **Michaelis-Menten** enzymes
consume it — and we solve one sinusoid **per GPU thread**. The output is the
single-pass **extraction ratio** (fraction of drug cleared) and how it varies with
blood velocity. It is a compact, verifiable teaching model of the microscopic basis
of hepatic clearance used in virtual pharmacology and organ-on-chip "digital twins".

## What this computes & why the GPU helps

Liver lobules and kidney nephrons are structurally repetitive functional units that
process blood to clear metabolites, drugs, and toxins. Simulating clearance across
millions of sinusoidal (liver) or tubular (kidney) segments enables virtual
pharmacotoxicology and organ-on-chip digital twins. Oxygen-zone-specific metabolism
(**periportal** vs. **centrilobular**) adds real physiological complexity, encoded
here as a spatially-varying enzyme capacity `Vmax(x)`.

**The parallel bottleneck:** each sinusoid requires an independent nonlinear ODE
solve (an RK4 march of `nseg` steps). A whole organ has *millions* of such
segments, and they do not interact — so the runtime is dominated by "solve the same
ODE for many parameter sets". The GPU maps this perfectly: **one thread integrates
one sinusoid**, turning a serial `O(nsin · nseg)` loop into a single parallel launch
(the ensemble-ODE pattern shared with flagships `9.02` and `13.02`).

## The algorithm in brief

- **Steady-state 1-D transport-reaction ODE** per sinusoid: `v·dC/dx = −Vmax(x)·C/(Km+C)`.
- **Michaelis-Menten clearance** — saturable enzyme kinetics (nonlinear → no closed
  form in general → integrate numerically).
- **Metabolic zonation** — `Vmax(x)` ramps linearly periportal → centrilobular.
- **RK4 in space** to march `C(x)` from inlet to outlet.
- **Ensemble over blood velocity** — `nsin` sinusoids swept from `v_lo` to `v_hi`.
- **Analytic first-order limit** for validation (exponential washout when `C ≪ Km`).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/liver-kidney-perfusion-modeling.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/liver-kidney-perfusion-modeling.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\liver-kidney-perfusion-modeling.sln /p:Configuration=Release /p:Platform=x64
```

No extra CUDA library is linked — the kernel is hand-written and needs only the CUDA
runtime (`cudart`).

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (uses the optional CMake build)
```

The demo builds if needed, runs on `data/sample/lobule.txt`, prints the per-sinusoid
result table and lobule summary, shows the GPU-vs-CPU agreement check and the
analytic cross-check, and prints a timing line on stderr.

## Data

- **Sample (committed):** `data/sample/lobule.txt` — a tiny, **synthetic** lobule
  config (4096 sinusoids) so the demo runs with zero downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print the provenance of the
  real public sources and (for credentialed sets) registration instructions only.
- **Provenance, field meanings & license:** see [data/README.md](data/README.md).

Real sources this model would be calibrated from: Human Protein Atlas liver
expression (https://www.proteinatlas.org); HMDB liver metabolomics (https://hmdb.ca);
Open Systems Pharmacology PBPK model library
(https://github.com/Open-Systems-Pharmacology/OSP-PBPK-Model-Library); PhysioNet
renal function datasets (https://physionet.org).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt). The
program computes the result on both the **GPU** (`src/kernels.cu`) and a **CPU
reference** (`src/reference_cpu.cpp`) that share the exact same physics
(`src/perfusion.h`), and asserts:

1. per-sinusoid GPU vs CPU agree to `≤ 1e-9` (observed `~2e-16`, round-off), and
2. the mean extraction ratio matches the analytic first-order limit to `≤ 1e-2`.

The clearance physics is visible in the table: slower blood (0.2 mm/s → 21.8%
extraction) is cleared far more than fast blood (1.0 mm/s → 4.8%), because it spends
longer next to the enzymes.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads the lobule, runs CPU + GPU, verifies (both
   ways), reports.
2. [`src/perfusion.h`](src/perfusion.h) — the **shared** `__host__ __device__`
   physics: Michaelis-Menten rate, zonation, RK4 spatial march.
3. [`src/reference_cpu.h`](src/reference_cpu.h) / [`.cpp`](src/reference_cpu.cpp) —
   the lobule config, the velocity sweep, and the trusted serial baseline.
4. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the thread-mapping idea.
5. [`src/kernels.cu`](src/kernels.cu) — the kernel (one thread per sinusoid) + host wrapper.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

- **Open Systems Pharmacology Suite** (https://github.com/Open-Systems-Pharmacology)
  — organ-level PBPK with liver/kidney compartments; study how single-pass
  extraction becomes an organ clearance term.
- **mrgsolve** (https://github.com/metrumresearchgroup/mrgsolve) — ODE-based organ
  pharmacokinetics; a clean many-ODE solver to compare structure with.
- **SimVascular `svFSI`** (https://github.com/SimVascular/svFSI) — vascular-tree
  flow for the portal vein; where a real `v(x)` would come from.
- **HemeLB** (https://github.com/hemelb-codes/hemelb) — lattice-Boltzmann
  microvessel flow for sinusoidal geometry (pairs with flagship `6.04`).

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Ensemble ODE integration** (PATTERNS.md §1): batch the same ODE across many
parameter sets, one **thread per sinusoid**, with the per-element physics in one
shared `__host__ __device__` header so CPU and GPU match to round-off. Custom CUDA
kernel for the zonal Michaelis-Menten reaction — no library needed for the
independent-tube teaching model. (The full *coupled* lobular network would add a
sparse linear solve via **cuSPARSE**, and the catalog's "hierarchical parallelism"
— blocks per lobule, threads per segment — is the natural multi-lobule extension;
both are discussed in THEORY §4 and §7.)

## Exercises

1. **Nonlinear zonation.** Replace the linear `Vmax(x)` in `perfusion.h` with an
   oxygen-driven sigmoid and observe how the extraction profile changes.
2. **Saturating dose.** Raise `C_in` toward and above `Km` (edit
   `data/sample/lobule.txt`) and watch the extraction ratio *fall* as the enzymes
   saturate — then explain why the analytic first-order cross-check degrades.
3. **Grid convergence.** Sweep `nseg` (10, 50, 200, 1000) and confirm the RK4
   result converges at `O(h⁴)`; find the smallest `nseg` within `1e-6` of the fine grid.
4. **Scale it.** Run `make_synthetic.py --nsin 1048576` and compare CPU vs GPU time
   as the ensemble grows — see the GPU advantage emerge past the launch-bound regime.
5. **Couple the network.** Add a shared portal-inflow node so sinusoid flows obey
   mass conservation, forming a sparse system — solve it with cuSPARSE (THEORY §7).

## Limitations & honesty

- **Reduced-scope teaching model.** One lobule; sinusoids are treated as
  **independent** plug-flow tubes with a *prescribed* velocity sweep. A real lobule
  is a **coupled** vascular network (flows obey mass conservation) with 3-D geometry
  and diffusion — omitted here for clarity (see THEORY §7 for the full picture).
- **Synthetic data.** Every number in `data/sample/lobule.txt` is invented but
  physiologically plausible; nothing here is calibrated to a real drug or patient.
- **No kidney sub-model yet.** The catalog spans liver *and* kidney; this teaching
  version implements the liver sinusoid. The nephron's filtration–reabsorption–
  secretion and medullary countercurrent use the same "thread per segment" mapping
  (an extension, described in THEORY §7).
- **Timing is a teaching artifact, not a benchmark.** The 4096-sinusoid sample is
  launch-bound; the GPU's edge grows with problem size.
- **Not for clinical use.** No output here may inform any medical decision.
