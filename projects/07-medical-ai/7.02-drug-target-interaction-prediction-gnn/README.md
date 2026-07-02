# 7.2 — Drug-Target Interaction Prediction (GNN)

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Medical%20AI%20%26%20Clinical%20Deep%20Learning-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 7: Medical AI & Clinical Deep Learning · Catalog ID `7.2`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

A drug works by binding a protein target; **Drug–Target Interaction (DTI)**
prediction asks a computer to guess which molecules bind which proteins so a
discovery pipeline can screen millions of candidates cheaply. Molecules are
**graphs** (atoms = nodes, bonds = edges), so we use a **graph neural network**:
each atom repeatedly gathers its neighbours' features along bonds, we pool those
into one embedding per drug, encode each protein into the same space, and score
every drug × protein pair with a dot product + sigmoid. This project is a
**reduced-scope teaching version** — a *fixed-weight (untrained)* forward pass —
that makes the CUDA data-flow of message passing and pairwise scoring explicit
and verifiable, without the training loop or protein transformers that a
production model adds.

## What this computes & why the GPU helps

Predicts whether a small molecule (drug) will bind a protein target and, in a
full system, its binding affinity (Kd/Ki) or a binary interaction label.
Molecular graphs have irregular topology, so graph message-passing aggregates
neighbour features **in parallel across many atoms and many candidate pairs at
once** on the GPU.

**The parallel bottleneck:** two stages dominate and both are embarrassingly
parallel. (1) **Message passing** is a *gather* over graph edges — one GPU thread
per atom sums its neighbours and applies a shared linear layer (no atomics: each
atom's output has a single writer). (2) **Pair scoring** is `D × P` independent
dot products (drug embedding · protein embedding); at virtual-screening scale
(`D` = millions of compounds, `P` = thousands of targets) this quadratic step is
where GPU throughput decides how many candidates you can score per day.

## The algorithm in brief

- **MPNN message passing** (`T` rounds): `h_i ← ReLU(W · Σ_{j∈N(i)} h_j + b)`,
  weights shared across atoms; self-loops added so a node keeps its own feature.
- **Graph-level sum pooling** → one embedding vector per drug ("readout").
- **Protein encoding**: one linear layer + ReLU into the shared embedding space.
- **DTI score**: `σ( (drug·protein) / F )` for every drug × protein pair.

Related methods (GAT, GIN, DMPNN, cross-attention protein encoders) are described
in [THEORY.md](THEORY.md), which has the full science → math → algorithm → GPU
mapping derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/drug-target-interaction-prediction-gnn.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/drug-target-interaction-prediction-gnn.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\drug-target-interaction-prediction-gnn.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/dti_sample.txt`, prints the DTI
score matrix and the top-ranked pair, shows the GPU-vs-CPU agreement check, and
prints a timing line.

## Data

- **Sample (committed):** `data/sample/dti_sample.txt` — a tiny, offline batch of
  6 synthetic drug graphs × 4 protein descriptors so the demo runs with zero
  downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print pointers to the real
  benchmarks (they never bypass credentials); `scripts/make_synthetic.py`
  regenerates or enlarges the synthetic sample.
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: **BindingDB** (~2.9M measured affinities,
<https://www.bindingdb.org/>), **ChEMBL** (>20M bioactivity records,
<https://www.ebi.ac.uk/chembl/>), **Davis** (442 kinases × 68 drugs), **KIBA**
(kinase-inhibitor benchmark).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt): a
`6×4` probability matrix, the top interaction `drug 5 <-> protein 3`, the note
`RECOVERED` (the model's top pair equals the implanted synthetic ground truth),
and `RESULT: PASS`. The program computes the forward pass on both the **GPU**
(`src/kernels.cu`) and a **CPU reference** (`src/reference_cpu.cpp`) and asserts
they agree to within `1e-4` (observed ~`1e-7`; the gap is fused-multiply-add
rounding — see THEORY §5–6). That agreement is the correctness guarantee.

## Code tour

Read in this order:

1. [`src/gnn.h`](src/gnn.h) — the shared `__host__ __device__` per-element math
   (linear layer, ReLU, dot, sigmoid); the source of CPU/GPU parity.
2. [`src/main.cu`](src/main.cu) — loads data, runs CPU + GPU, verifies, reports.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the thread-mapping idea.
4. [`src/kernels.cu`](src/kernels.cu) — the four kernels (message pass / pool /
   encode / score) and the host wrapper.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the loader, the fixed
   (seeded) weights, and the trusted serial baseline.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

- **DeepPurpose** (<https://github.com/kexinhuang12345/DeepPurpose>) — 15
  drug/protein encoders, 50+ DTI architectures; learn how encoders slot into a
  shared scoring head.
- **TorchDrug** (<https://github.com/DeepGraphLearning/torchdrug>) — GPU graph
  learning; learn batched-graph engineering.
- **DGL-LifeSci** (<https://github.com/awslabs/dgl-lifesci>) — molecular GNN
  toolkit with CUDA-backed sparse ops (the SpMM our gather approximates).
- **DTA-GNN** (<https://github.com/lennylv/DTA-GNN>) — target-specific
  drug–target-affinity dataset construction and GNN training.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Gather over graph edges** (one thread per node, CSR indirection, ping-pong
double buffering across rounds) + **independent-jobs pairwise scoring** (one
thread per drug × protein pair), with the tiny shared weights in **constant
memory** (broadcast cache). This mirrors `docs/PATTERNS.md §1` ("gather" and
"independent jobs") and the constant-memory idiom of flagship 1.12. Production
systems (DGL/PyG, cuSPARSE SpMM, cuBLAS GEMM, cuDNN, Flash Attention) implement
these same shapes at scale — see THEORY §7.

## Exercises

1. **Add attention (GAT).** Weight each neighbour by a learned/α-scored
   coefficient instead of a plain sum in `message_pass_kernel`, and renormalize.
2. **Mean vs sum readout.** Change `pool_kernel` to average nodes (divide by
   `n_d`); does the recovered top pair change? Why?
3. **Scale it up.** `python scripts/make_synthetic.py --drugs 4096 --proteins 512`,
   then compare CPU vs GPU timing — watch the GPU's edge appear once the `D×P`
   score matrix is large (the demo's tiny batch is launch-bound; THEORY §3).
4. **Tile the score step.** Load a block of drug embeddings into shared memory and
   reuse them across proteins — the first step toward a GEMM (`Z Yᵀ`).
5. **FP64 variant.** Template the math on `float`/`double` and confirm the
   CPU/GPU error shrinks below `1e-12` (the FMA gap is precision-relative).

## Limitations & honesty

- **The network is UNTRAINED.** Weights are deterministically seeded (`gnn.h`,
  `reference_cpu.cpp`), so the scores demonstrate the *machinery*, not real
  binding — they carry **no clinical meaning** and must never inform a decision.
- **Reduced scope.** No training loop, no GAT/GIN/DMPNN, no protein transformer,
  no Flash Attention, no batched cuBLAS/cuSPARSE — those are described in
  THEORY §7 as the production path.
- **Synthetic data.** The sample is small hand-built chain graphs and descriptor
  vectors, labeled synthetic everywhere; the "ground truth" is defined as the pair
  the fixed model ranks highest, so recovery validates the pipeline, not chemistry.
- **Timing is a teaching artifact**, not a benchmark: the tiny batch is dominated
  by launch/copy overhead (THEORY §3, PATTERNS §7).
