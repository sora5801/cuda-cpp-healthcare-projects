# 1.19 — Network / Polypharmacology Modeling

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Drug%20Discovery%20%26%20Molecular%20Design-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 1: Drug Discovery & Molecular Design · Catalog ID `1.19`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

Most drugs hit **more than one protein** — that is *polypharmacology*, and it is
both how side effects arise and how a drug can be repurposed for a new disease.
A standard way to predict "which other proteins does this drug bind?" is to build
a **knowledge graph** of drugs, proteins, and their relationships, embed every
entity and relation as a vector with **TransE**, and then ask: under the rule
`drug + TARGETS ≈ protein`, which proteins does this drug land near? This project
implements that **link-prediction scoring step on the GPU**: one thread scores the
query drug against one candidate protein, all candidates in parallel, then we rank
them. On a synthetic graph with a baked-in answer, the demo recovers exactly the
proteins it was meant to find.

## What this computes & why the GPU helps

Polypharmacology recognizes that drugs interact with multiple targets, creating
complex biological networks. GPU-accelerated graph neural networks on drug-target
interaction (DTI) networks, protein-protein interaction (PPI) networks, and
disease-gene networks enable systems-level prediction of off-target effects, drug
combinations, and drug repurposing. Large-scale heterogeneous graph training
(heterogeneous GNN, knowledge graph embeddings) with millions of nodes requires
GPU memory and compute.

**The parallel bottleneck:** link prediction scores a query `(head, relation)`
against **every candidate tail entity** in the graph — for a real knowledge graph
that is tens of thousands of proteins (or millions of entities for a full
biomedical KG). Each candidate's score is an **independent** `d`-dimensional
distance computation, so the scoring is *embarrassingly parallel*: we give each
candidate tail its own GPU thread. This is the same "one independent job per
thread" pattern as the Tanimoto flagship (`1.12`), and it is the step that
dominates inference-time ranking.

## The algorithm in brief

- **TransE knowledge-graph embedding.** A fact is a triple `(head h, relation r,
  tail t)`; TransE trains embeddings so that true facts satisfy `h + r ≈ t`.
- **Link-prediction scoring.** The plausibility of a candidate tail is the
  **negative L2 distance** `score(t) = −‖(h + r) − t‖`. We use the negative
  *squared* distance (monotonic, no `sqrt`, exact arithmetic) for ranking.
- **Top-K ranking + recovery.** Sort candidates by score; report the top-K
  predicted targets and how many ground-truth targets they recovered.
- Related methods named in the catalog and discussed in `THEORY.md`: RotatE
  (rotations in complex space), GraphDTA/DeepDTA (GNN/CNN DTI predictors),
  network diffusion, community detection, DeepSynergy (drug-combination synergy).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/network-polypharmacology-modeling.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/network-polypharmacology-modeling.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\network-polypharmacology-modeling.sln /p:Configuration=Release /p:Platform=x64
```

This project links only the CUDA runtime (`cudart_static.lib`); it uses constant
memory and a plain compute kernel, no extra CUDA libraries.

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/`, prints the top-5 predicted
targets and the recovery metric, shows the GPU-vs-CPU agreement check, and prints
a timing line.

## Data

- **Sample (committed):** `data/sample/kg_embeddings_sample.txt` — a tiny,
  **synthetic** TransE knowledge graph (1 drug, 1 relation, 64 protein tails,
  dim 16) with a recoverable known answer, so the demo runs offline with zero
  downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print where to obtain the
  real graphs (they do not auto-download credentialed/non-redistributable data).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: STRING PPI network (https://string-db.org); DrugBank —
FDA-approved drugs and targets (https://go.drugbank.com); STITCH — drug-protein
interactions (http://stitch.embl.de); DrugComb — drug combination synergy data
(https://drugcomb.fimm.fi).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt):

```
1.19 -- Network / Polypharmacology Modeling
TransE link prediction: 1 query drug vs 64 protein tails (dim=16)
top-5 predicted targets (protein index : TransE score):
  #1  protein[6]  score = -0.000000
  #2  protein[15]  score = -0.002517
  #3  protein[62]  score = -0.008118
  #4  protein[30]  score = -4.857530
  #5  protein[52]  score = -5.265568
recovery: 3 / 3 ground-truth targets in top-5
RESULT: PASS (GPU matches CPU within tol=1.0e-05)
```

The program computes the scores on both the **GPU** (`src/kernels.cu`) and a
**CPU reference** (`src/reference_cpu.cpp`) and asserts they agree within tolerance
— that agreement is the correctness guarantee. The tolerance is `1e-5` rather than
`0` because the GPU fuses multiply-add (FMA) where the host compiler does not; the
divergence (~`1e-7`) is real and explained in `THEORY.md` and `demo/README.md`.

## Code tour

Read in this order:

1. [`src/transe.h`](src/transe.h) — the `__host__ __device__` scoring core
   (`h + r − t`), shared by CPU and GPU so their numbers match.
2. [`src/main.cu`](src/main.cu) — loads the graph, runs CPU + GPU, verifies,
   ranks, and reports.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the
   one-thread-per-tail idea + the constant-memory query.
4. [`src/kernels.cu`](src/kernels.cu) — the kernel and host wrapper.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the loader and the trusted
   serial baseline.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

- **PyTorch Geometric** (https://github.com/pyg-team/pytorch_geometric) — GPU
  heterogeneous graph learning; the reference toolkit for training real TransE /
  RotatE / GNN embeddings whose output this project would consume.
- **DGL** (https://github.com/dmlc/dgl) — GPU graph learning for DTI networks;
  good for the message-passing GNN variants named in the catalog.
- **DeepPurpose** (https://github.com/kexinhuang12345/DeepPurpose) — a
  drug-target interaction prediction toolkit (DeepDTA/GraphDTA); study it for how
  DTI is framed end to end.
- **OpenKE** (https://github.com/thunlp/OpenKE) — a focused library of knowledge
  graph embeddings (TransE, RotatE, …); the cleanest place to see the training
  loop this project's scoring step assumes.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Independent jobs + constant-memory query** (PATTERNS.md §1; exemplar `1.12`).
Each candidate tail is scored by one thread via a grid-stride loop; the shared
query head and relation vectors live in `__constant__` memory so the constant
cache broadcasts them warp-wide instead of re-reading them from global memory per
thread. The per-tail math lives in a single `__host__ __device__` header
(`transe.h`, PATTERNS.md §2) so the CPU reference and the GPU kernel run identical
arithmetic. (The catalog also mentions cuSPARSE adjacency products and FP16
embedding tables for the *training* side; those belong to the full GNN pipeline
described in `THEORY.md` "Where this sits in the real world".)

## Exercises

1. **RotatE.** Replace the translational score `‖h + r − t‖` with RotatE's
   `‖h ∘ r − t‖` (element-wise complex rotation). How does the kernel change?
   (Hint: store embeddings as interleaved real/imaginary pairs.)
2. **Filtered ranking.** Real evaluation *filters* out other known-true tails
   before ranking a held-out one. Add a known-positives mask and recompute the
   rank of a held-out target (Hits@K, mean reciprocal rank).
3. **Bigger graph.** Generate `--n 100000 --dim 64` with `make_synthetic.py` and
   watch the GPU-vs-CPU timing gap grow (the tiny sample is launch-bound).
4. **Shared-memory staging.** For very large `dim`, stage the query into shared
   memory once per block instead of reading constant memory per dimension; measure
   the difference.
5. **Negative sampling.** Implement GPU-batched negative sampling (corrupt the
   tail of true triples) — the core of the *training* loop that produces these
   embeddings.

## Limitations & honesty

- **Reduced-scope teaching version.** The research-grade task (catalog "Active
  R&D") is *training* heterogeneous GNN / KG embeddings on million-node graphs.
  This project ships **pre-computed synthetic embeddings** and implements the
  **scoring + ranking** step on the GPU. The full training pipeline (negative
  sampling, margin-ranking loss, SGD, cuSPARSE message passing) is described in
  `THEORY.md` but not implemented here.
- **Synthetic data.** The committed graph is generated with a known answer baked
  in; it is **not** real biology and no index maps to a real drug or protein.
- **TransE is a simple model.** It cannot represent one-to-many or symmetric
  relations well — RotatE and GNNs exist precisely to fix that (`THEORY.md`).
- **Not for clinical use.** Nothing here is a real off-target prediction; this is
  study material only.
