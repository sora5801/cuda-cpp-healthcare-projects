# 6.9 — Agent-Based Tissue / Immune Simulation

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Computational%20Physiology%20%26%20Systems%20Biology-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 6: Computational Physiology & Systems Biology · Catalog ID `6.9`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

A **reduced-scope, GPU-accelerated agent-based model (ABM)** of tumor–immune
tissue, in the spirit of [PhysiCell](https://github.com/MathCancer/PhysiCell).
Tissue is a **population of autonomous cell agents**: tumor cells secrete a
chemokine into a diffusing substrate field, and immune cells **chemotax** up that
gradient toward the tumor while pushing off any neighbour they overlap. Each
timestep couples three GPU kernels — an **atomic scatter** (secretion), a
**stencil PDE** (diffusion), and a **spatial-binning** neighbour search
(mechanics + chemotaxis) — so it is a compact tour of three canonical GPU
patterns at once. The demo shows immune cells migrating inward (mean
immune→tumor distance drops from ≈11.4 to ≈8.0).

## What this computes & why the GPU helps

An ABM tracks each cell's position, type, and interactions individually. Two
independent costs dominate and both are GPU-friendly:

- **Cell–cell mechanics** is a pairwise neighbour search. All-pairs is **O(N²)**;
  hashing cells into a uniform bin grid and only testing the **3×3 neighbouring
  bins** drops it to **O(N)** — one thread per cell. This is the ABM-specific
  acceleration (PhysiCell scales to 10⁵–10⁶ cells this way).
- **Substrate (chemokine) diffusion** is a nearest-neighbour **stencil** over a
  Cartesian grid — one thread per grid cell, no global communication.

**The parallel bottleneck** is therefore twofold: the per-cell force/chemotaxis
update and the per-grid-cell diffusion, every timestep. We give each its own
thread grid and iterate.

## The algorithm in brief

Per timestep (secrete → diffuse → move):

- **Secrete** — each tumor cell `atomicAdd`s a **fixed-point quantum** of
  chemokine into the grid cell it occupies (integer atomics → deterministic,
  CPU-exact).
- **Diffuse** — one explicit forward-Euler step of `∂c/∂t = D∇²c − decay·c`
  (5-point Laplacian, zero-flux walls), ping-ponging two field buffers.
- **Move** — build spatial bins; each cell sums soft-sphere repulsion from
  overlapping neighbours (found via the bins) and, if immune, adds a chemotactic
  velocity `= chemotaxis · ∇c`; integrate positions by overdamped forward Euler.

See [THEORY.md](THEORY.md) for the equations, complexity, and the full GPU mapping.

## Build

Requires **Visual Studio 2026** (v145) + **CUDA 13.3** ([docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/agent-based-tissue-immune-simulation.sln`.
2. Select **`Release|x64`** → **Build** → `build/x64/Release/agent-based-tissue-immune-simulation.exe`.

CLI: `msbuild build\agent-based-tissue-immune-simulation.sln /p:Configuration=Release /p:Platform=x64`

## Run the demo

```powershell
./demo/run_demo.ps1
```

Runs the tumor–immune scenario on CPU + GPU and verifies they agree.

## Data

- **Sample (committed):** `data/sample/tissue_params.txt` — the one-line scenario
  (grid, timing, PDE coefficients, cell counts, RNG seed). Cell positions are
  generated **deterministically** from the seed inside the loader.
- **No download needed** to run the demo. Real single-cell / immune-landscape
  datasets for *calibration* are listed in [data/README.md](data/README.md) and
  `scripts/download_data.ps1` (credentialed; not auto-fetched).
- Bigger synthetic scenario:
  `python scripts/make_synthetic.py --gx 64 --gy 64 --n-tumor 400 --n-immune 300 --steps 800`.

## Expected output

`demo/expected_output.txt` holds the deterministic summary. The chemokine field
**total** is summed in integer fixed-point quanta, so the GPU (`src/kernels.cu`)
and CPU (`src/reference_cpu.cpp`) — which share the per-element physics in
`src/abm_core.h` — agree on it **exactly** (`901229917` quanta for the sample).
Final cell positions agree within `1e-6` domain units (the tiny residual is the
GPU's fused-multiply-add differing from the host compiler over 300 steps — see
THEORY §Numerics). The peak chemokine sits over the tumor at the domain centre,
and the mean immune→tumor distance shrinks — the science check that chemotaxis works.

## Code tour

1. [`src/main.cu`](src/main.cu) — load the scenario, run CPU + GPU, verify, print the summary.
2. [`src/abm_core.h`](src/abm_core.h) — **the shared `__host__ __device__` physics**: fixed-point quanta, the diffusion stencil, the gradient, and the per-cell move (repulsion + chemotaxis).
3. [`src/reference_cpu.h`](src/reference_cpu.h) / [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — loader, deterministic cell placement, the **spatial-binning counting sort**, the serial reference, and the summary.
4. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface and the three-pattern teaching idea.
5. [`src/kernels.cu`](src/kernels.cu) — the four kernels (secrete/fold/diffuse/move) + the host time loop.

## Prior art & further reading

- **PhysiCell** (<https://github.com/MathCancer/PhysiCell>) — 3-D multicellular simulator with center-based mechanics + biotransport; the reference for this pattern. Study its diffusion (BioFVM) and cell-cycle models.
- **PhysiBoSS** (<https://github.com/PhysiBoSS/PhysiBoSS>) — couples Boolean intracellular signaling (MaBoSS) to PhysiCell; shows how to add per-cell decision logic.
- **Chaste** (<https://github.com/Chaste/Chaste>) — off-lattice cell-based models with vertex/Voronoi mechanics; a different mechanical formulation to compare.
- **MOOSE** (<https://github.com/BhallaLab/moose-core>) — chemical signaling *within* cells; complements the tissue-scale view here.

Study these for production ABM; this project reimplements the core hybrid pattern didactically (CLAUDE.md §2), not by copying code.

## CUDA pattern used here

A **hybrid** of three flagship patterns in one loop: **atomic scatter-reduction**
with deterministic fixed-point integers (secretion; cf. 11.09/5.01) · a
nearest-neighbour **stencil** with ping-pong buffers (diffusion; cf. 6.04/14.02)
· **spatial binning** for O(N) neighbour search (mechanics + chemotaxis; the
ABM-specific idiom) · a shared `__host__ __device__` core for exact CPU/GPU parity.

## Exercises

1. **On-GPU binning.** Replace the host bin rebuild with a device counting sort
   (Thrust `sort_by_key` on the cell→bin keys) so the whole step stays on the GPU
   — the biggest performance lever here (see THEORY §GPU-mapping).
2. **Cell cycle / proliferation.** Give tumor cells a Ki67 cycle: they grow and
   divide (append a new agent) at a rate that drops where chemokine or crowding
   is high. Watch the nodule expand.
3. **Immune killing.** When an immune cell is within contact of a tumor cell for
   `k` steps, remove the tumor cell. Track the tumor count over time.
4. **Adhesion.** Add a short-range *attractive* term (not just repulsion) so like
   cells stick — the second half of PhysiCell's center-based mechanics.
5. **Scale it.** Grow to `--n-tumor 4000 --n-immune 3000 --gx 128 --gy 128` and
   measure where the GPU overtakes the CPU (with Exercise 1 done).

## Limitations & honesty

- **Reduced-scope 2-D teaching version.** Real ABMs (PhysiCell) are 3-D, with
  multiple substrates, full cell-cycle/death models, adhesion, and secretion +
  uptake kinetics. Here: one chemokine, repulsion-only mechanics, chemotaxis, no
  proliferation or death.
- **Synthetic scenario.** The cell layout and parameters are synthetic (labelled
  as such); this is **not** a validated tumor–immune model and is **not for
  clinical use**.
- **Hybrid binning.** The spatial bins are rebuilt on the host each step and
  re-uploaded — a deliberate simplification for exact CPU/GPU parity and clean
  teaching; it makes the demo **launch-bound** on tiny inputs (the GPU is *slower*
  than the CPU here). Exercise 1 removes this bottleneck.
- **Explicit diffusion** requires `D·dt/dx² ≤ 0.25`; the loader refuses unstable
  configs. Production BioFVM uses an implicit ADI (Thomas) solver with no such limit.
