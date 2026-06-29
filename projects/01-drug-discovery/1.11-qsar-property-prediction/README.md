# 1.11 — QSAR / Property Prediction

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Drug%20Discovery%20%26%20Molecular%20Design-lightgrey)

> **🟢 Beginner · Established** — Domain 1: Drug Discovery & Molecular Design · Catalog ID `1.11`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

A **QSAR** (Quantitative Structure–Activity Relationship) model predicts a
molecule's property — solubility, toxicity, binding affinity — directly from its
structure. This project implements the modern, graph-based way to do that: a
**Graph Convolutional Network (GCN)**, the simplest message-passing neural
network. Atoms are graph nodes, bonds are edges; each GCN layer lets every atom
mix in its neighbors' features, and after two layers a mean-pool readout + linear
head turns each molecule into one predicted number. We run the whole batch on the
GPU (one thread per atom) and check it against a CPU reference. The model is small
and **untrained on purpose** — the lesson is the GPU *message-passing pattern* and
exact CPU↔GPU agreement, not a real chemical prediction.

## What this computes & why the GPU helps

Quantitative structure-activity relationship (QSAR) models predict biological
activity from molecular descriptors or learned representations. Modern approaches
use message-passing neural networks (MPNNs) over molecular graphs, enabling
GPU-batched inference/training on millions of labeled datapoints. The bottleneck
is **message aggregation over irregular graph structures** — gathering and summing
each atom's neighbors — which PyTorch Geometric or DGL run on CUDA backends.
GPU-accelerated QSAR models at pharmaceutical companies screen hundreds of
millions of virtual compounds per hour for ADMET and activity filters.

**The parallel bottleneck:** the per-atom **neighbor aggregation** inside each GCN
layer. Every atom independently gathers its bonded neighbors, projects them through
the weight matrix, and sums — `O(E · F · F')` work that is embarrassingly parallel
across atoms (and across molecules). We give **each output atom its own GPU
thread** so a whole layer is one kernel launch.

## The algorithm in brief

- **Graph convolution** (Kipf & Welling): `H' = ReLU( D̃^{-1/2}(A+I)D̃^{-1/2} · H · W )`
  — symmetric-normalized neighbor aggregation + a learned linear map + ReLU.
- **Two stacked layers** (F_IN=6 → F_HID=8 → F_OUT=4) so information flows 2 hops.
- **Readout**: permutation-invariant mean pooling over each molecule's atoms.
- **Head**: a linear layer maps the pooled embedding to one scalar property.
- Related production variants (D-MPNN/Chemprop, GAT, Uni-Mol, RF/XGBoost on Morgan
  fingerprints, ensemble/MC-Dropout uncertainty) are surveyed in THEORY §7.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/qsar-property-prediction.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/qsar-property-prediction.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\qsar-property-prediction.sln /p:Configuration=Release /p:Platform=x64
```

The project links only `cudart_static.lib` (the CUDA runtime) — no extra CUDA
libraries, because the message passing is hand-rolled to stay a clear teaching
surface (no black boxes).

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (CMake build)
```

The demo builds if needed, runs on `data/sample/molecules_sample.txt`, prints each
molecule's predicted property and the top-ranked molecule, shows the GPU-vs-CPU
agreement check, and prints a timing line on stderr.

## Data

- **Sample (committed):** `data/sample/molecules_sample.txt` — a tiny, **synthetic**
  batch of 5 toy molecules (23 atoms) in batched-CSR format, so the demo runs
  offline with zero downloads.
- **Full datasets:** `scripts/download_data.ps1` / `.sh` print where to get the real
  benchmarks and how to featurize them.
- **Provenance, file format & field meanings:** see [data/README.md](data/README.md).

Catalog dataset notes: MoleculeNet — curated ML benchmark for 17+ molecular
datasets (https://moleculenet.org); ChEMBL bioactivity data
(https://www.ebi.ac.uk/chembl/); TDC (Therapeutics Data Commons) — 66 tasks for
drug discovery ML (https://tdcommons.ai); PCBA (PubChem BioAssay) — 128 bioassays
on 440k compounds (https://moleculenet.org).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt):

```
1.11 -- QSAR / Property Prediction
GCN inference: 5 molecules, 23 atoms total (F_IN=6, F_HID=8, F_OUT=4)
  mol  0 ( 4 atoms): predicted property = -0.582784
  mol  1 ( 6 atoms): predicted property = -0.651793
  mol  2 ( 4 atoms): predicted property = -0.554825
  mol  3 ( 4 atoms): predicted property = -0.536602
  mol  4 ( 5 atoms): predicted property = -0.641983
top-ranked molecule: mol 3 (property = -0.536602)
RESULT: PASS (GPU predictions match CPU within 1e-04)
```

The program computes the predictions on both the **GPU** (`src/kernels.cu`) and a
**CPU reference** (`src/reference_cpu.cpp`) and asserts they agree within `1e-4`.
Because both call the same `gcn.h` math in the same neighbor order, the measured
gap is ~`6e-8` (printed on stderr) — pure fp32 FMA rounding. That agreement is the
correctness guarantee.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads the batch, runs CPU + GPU, verifies, reports.
2. [`src/gcn.h`](src/gcn.h) — the shared `__host__ __device__` per-node math
   (aggregation + transform + ReLU + readout). The heart of the project.
3. [`src/reference_cpu.h`](src/reference_cpu.h) / [`.cpp`](src/reference_cpu.cpp) —
   the CSR data model, the loader, the seeded weights, and the serial reference.
4. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the thread-mapping idea.
5. [`src/kernels.cu`](src/kernels.cu) — the two layer kernels, the readout kernel,
   and the host wrapper (constant-memory weights, three launches).
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

- **Chemprop** (https://github.com/chemprop/chemprop) — D-MPNN for molecular
  property prediction with GPU training; *the* reference for message-passing QSAR.
- **Uni-Mol** (https://github.com/deepmodeling/Uni-Mol) — 3-D molecular transformer
  pre-trained on 209M conformers; what to read for geometry-aware models.
- **DeepChem** (https://github.com/deepchem/deepchem) — broad GPU-accelerated ML
  chemistry toolkit; good for end-to-end pipelines and featurizers.
- **DGL-LifeSci** (https://github.com/awslabs/dgl-lifesci) — GNNs for life science;
  study its batched-graph CSR handling, which mirrors our `Graph` layout.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Per-output gather over a CSR graph** (docs/PATTERNS.md §1): one thread per output
atom pulls and sums its neighbor list, so there are no atomics and the GPU sums in
the same order as the CPU (deterministic, exact agreement). The tiny weight tensors
live in **constant memory** (broadcast cache) since every thread reads them
unchanged — the same idea as the query in flagship 1.12. The per-node arithmetic is
a shared `__host__ __device__` core (PATTERNS.md §2) so CPU and GPU run identical
math. (Production stacks add PyTorch Geometric sparse ops, cuDNN dense layers, and
FP16 mixed precision — surveyed in THEORY §7.)

## Exercises

1. **Train it.** Replace the seeded weights with ones learned by gradient descent
   against a synthetic target (e.g. "predict atom count" or a logP-like sum), and
   watch the predictions become meaningful. Add an MSE loss + finite-difference or
   autograd gradients.
2. **Add a third layer / change widths.** Bump `GCN_F_HID`/`GCN_F_OUT` in `gcn.h`
   and add a layer; observe how 3 hops change the predictions. (Mind the constant
   `float row[GCN_F_HID]` staging buffer.)
3. **Swap mean pooling for sum or max pooling** in `gcn_readout_head`; explain why
   sum pooling makes the prediction grow with molecule size.
4. **Scale up.** Extend `make_synthetic.py` to emit 10⁵ molecules and re-time:
   confirm the GPU's per-kernel time grows far slower than the CPU's, recovering
   the speed-up that the tiny demo hides (PATTERNS.md §7).
5. **GAT-style attention.** Replace the fixed `c_ij = 1/sqrt(deg_i deg_j)` with a
   learned attention weight; compare to the symmetric-normalization baseline.

## Limitations & honesty

- **Untrained, synthetic.** The weights come from a seeded generator and the
  molecules are toy graphs — the predicted "property" is a demonstration number
  with **no chemical meaning** and **no clinical or design validity**.
- **Inference only.** No backpropagation/training loop (described in THEORY §7).
- **Toy featurization.** 6 hand-made atom features vs RDKit's dozens; no bond
  features, no 3-D geometry, no aromaticity perception.
- **Reduced scope.** A real QSAR MPNN (D-MPNN, GAT, graph transformer) is wider,
  deeper, and trained on measured labels; this is the smallest model that still
  teaches the full message-passing-on-GPU pattern.
- **Launch-bound timing.** On a 23-atom batch the GPU is *slower* than the CPU; the
  GPU's advantage only appears at screening scale (10⁵–10⁸ molecules). Timings are
  a teaching artifact, never a benchmark claim.
