# 1.15 — Protein-Ligand Binding Affinity Scoring (ML)

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Drug%20Discovery%20%26%20Molecular%20Design-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 1: Drug Discovery & Molecular Design · Catalog ID `1.15`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

A *scoring function* answers the central question of virtual drug screening: **how
tightly does this small molecule (the ligand) bind this protein in this docked
pose?** Classical scoring uses a physics force field; the modern machine-learning
approach instead **learns** the structure → affinity map from thousands of measured
complexes. This project implements the canonical **3D-CNN scorer** as a small,
fully-explicit GPU forward pass: it voxelizes a protein-ligand complex into a 3D
atomic-density grid, runs a 3D convolution + ReLU, global-average-pools, and reads
out a single predicted **pKd** (binding affinity). It scores a whole **batch** of
poses at once — exactly the "rescore millions of docking poses" workload that makes
GPUs indispensable in screening. Every multiply-add is visible and verified against
a CPU reference; the network weights are a deterministic stand-in for a trained
model, because here we teach the **GPU inference pattern**, not training.

## What this computes & why the GPU helps

End-to-end ML scoring functions learn protein-ligand interaction energy surrogates
directly from structural data, bypassing physics-based force fields. Models range
from 3D-CNNs over voxelized complexes to equivariant GNNs over atom graphs to
transformer co-folding models. GPU inference enables rapid rescoring of millions of
docking poses in virtual screening — a 3D-CNN scores a pose in ~1 ms on a GPU vs.
>1 s for a free-energy-perturbation calculation. The fundamental challenge is
generalization across chemical space and protein families.

**The parallel bottleneck:** post-docking rescoring evaluates the *same* network on
*millions of independent poses*. Two nested parallelisms make this a textbook GPU
job: (1) **across poses** — each pose's forward pass is independent, so we give each
pose its own thread block; and (2) **within a pose** — the 3D convolution is a
per-output-voxel stencil over a `16³` grid, so the block's threads cooperate over
voxels. The arithmetic is dense and regular (multiply-accumulate over a `3³`
neighborhood × 8 input channels × 8 filters per voxel), which is exactly what a GPU's
thousands of FMA units are built for.

## The algorithm in brief

- **Voxelize:** splat each atom as a Gaussian density blob into an `[8][16³]` grid;
  the 8 input channels are {C, N, O, S} × {protein, ligand} (atom species × side).
- **Conv3D + ReLU:** 8 learned filters, each `8 × 3³` taps, slide over the grid
  (SAME padding) → 8 feature maps, rectified.
- **Global average pool:** average each feature map's `16³` voxels → an 8-vector.
- **Dense readout:** weighted sum + bias → a raw score, logistic-squashed into a
  plausible **pKd ∈ [2, 11]**.
- **Verify:** the GPU result must match a serial CPU forward pass within tolerance.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/protein-ligand-binding-affinity-scoring-ml.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/protein-ligand-binding-affinity-scoring-ml.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\protein-ligand-binding-affinity-scoring-ml.sln /p:Configuration=Release /p:Platform=x64
```

Only the CUDA runtime (`cudart`) is linked — no extra CUDA library is needed, because
the conv/pool/dense math is hand-written here on purpose (the point is to *see* it).

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (via the optional CMake build)
```

The demo builds if needed, runs on `data/sample/complexes_sample.txt`, prints the
per-complex predicted pKd table and the rank-1 binder, shows the GPU-vs-CPU agreement
check, and prints a timing line (kernel vs CPU reference).

## Data

- **Sample (committed):** `data/sample/complexes_sample.txt` — 6 **synthetic** docked
  protein-ligand poses (~200 atoms total), so the demo runs offline with zero
  downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print instructions for the
  real benchmarks (they require registration and are not auto-downloaded).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: **PDBbind v2020** — 19,443 protein-ligand complexes with Kd/Ki
(<http://www.pdbbind.org.cn>); **CASF-2016** benchmark
(<http://www.pdbbind.org.cn/casf.php>); **ChEMBL** activity data
(<https://www.ebi.ac.uk/chembl/>); **BindingDB** — 2.8M measured binding affinities
(<https://www.bindingdb.org>).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt): a table of
6 predicted pKd values, the rank-1 predicted binder, and `RESULT: PASS`. The program
computes the score on both the **GPU** (`src/kernels.cu`) and a **CPU reference**
(`src/reference_cpu.cpp`) and asserts they agree within `1e-6` pKd — that agreement is
the correctness guarantee. (On the reference machine the two differed by only
`~2e-15`, essentially machine precision; the tolerance exists because the GPU pools
with a tree reduction and the CPU with a flat sum — see THEORY.)

## Code tour

Read in this order:

1. [`src/scoring_core.h`](src/scoring_core.h) — the **shared** per-element math
   (voxelization, conv tap, ReLU, weight generator) that BOTH CPU and GPU call.
2. [`src/main.cu`](src/main.cu) — loads data, runs CPU + GPU, verifies, reports.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the two-level
   (batch × stencil) thread-mapping idea.
4. [`src/kernels.cu`](src/kernels.cu) — the batched scoring kernel and host wrapper.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline + loader.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

- **GNINA** (<https://github.com/gnina/gnina>) — production CNN rescoring inside a
  docking pipeline; the source of the "Gaussian atom gridding" we mimic.
- **DeepChem** (<https://github.com/deepchem/deepchem>) — `AtomicConvolutions` and
  MPNN-based scorers; a great read for the graph-based alternative to voxel grids.
- **NeuralPLexer** (<https://github.com/zrqiao/NeuralPLexer>) — state-specific
  co-folding that also predicts affinity (the transformer/diffusion frontier).
- **DiffDock** (<https://github.com/gcorso/DiffDock>) — generative docking with an
  affinity proxy.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Batched independent jobs (one block per pose) × stencil (one thread per output
voxel).** The catalog lists "cuDNN for 3D-CNN layers; PyTorch Geometric for
equivariant message passing; FP16 mixed precision; GPU-parallel batch scoring for
post-docking rescoring of millions of poses." We hand-roll the conv/pool/dense in
double precision so the learner can see exactly what cuDNN does under the hood, and we
verify it exactly — see THEORY "Where this sits in the real world" for what changes in
a production (cuDNN + FP16 + streamed) build.

## Exercises

1. **Bigger batch, real timing.** Run `python scripts/make_synthetic.py --n 100000`,
   point the demo at the new file, and watch the GPU/CPU timing ratio flip in the
   GPU's favor as launch overhead is amortized. Plot kernel-ms vs `n`.
2. **Add a second conv layer.** Insert a conv→ReLU between the first conv and the
   pool (you'll need a second grid buffer). What changes in the verification
   tolerance, and why?
3. **Shared-memory tiling.** The conv currently re-reads the grid from global memory
   for every output voxel. Tile a `(BLOCK+2)³` halo region into shared memory and
   measure the bandwidth win (PATTERNS.md §1 "shared-memory tiling + halo").
4. **FP16 inference.** Switch the grid + weights to `__half` and compare accuracy and
   speed. Where does the squashed pKd diverge, and is it within a chemically
   meaningful tolerance?
5. **Real weights.** Replace `lcg_weight()` with weights loaded from a file and load a
   small trained 3D-CNN (e.g. exported from a PyTorch toy model). Does the rank-1
   binder change?

## Limitations & honesty

- **The model is untrained.** Weights come from a deterministic LCG, not from fitting
  PDBbind. The predicted pKd values are therefore **not chemically meaningful** — they
  exist only to exercise and *verify* the GPU forward pass. Do not read affinity into
  them. A real scorer loads trained weights (see THEORY).
- **The data is synthetic.** The committed sample is generated by
  `scripts/make_synthetic.py` and labeled synthetic everywhere. The `synthetic_label`
  column is a toy "true pKd" the model never sees.
- **Reduced scope.** One conv layer, `16³` grid, 4 atom types, double precision for
  exact verifiability. Production scorers use ~`24³` grids, several conv blocks,
  richer atom features (charge, hybridization, hydrogen-bonding), FP16, and proper
  pocket centering. THEORY spells out the gap.
- **Not for clinical or chemical use.** This is study material (CLAUDE.md §1, §8).
