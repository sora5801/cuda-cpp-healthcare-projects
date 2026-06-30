# 3.18 — Protein Language Model Inference

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Genomics%2C%20Sequencing%20%26%20Bioinformatics-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 3: Genomics, Sequencing & Bioinformatics · Catalog ID `3.18`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

A **protein language model** (PLM) like Meta's **ESM-2** reads an amino-acid
sequence and, layer by layer, lets every residue "look at" every other residue to
build context-aware embeddings that encode structure and function. The engine of
that looking is **multi-head self-attention**. This project implements **one
self-attention block** — embeddings → Q/K/V projections → `softmax(QKᵀ/√d)·V` per
head → output projection — on the GPU at a size a learner can trace by hand, and
verifies it against a plain-C++ reference. It teaches the single most important
GPU workload in modern AI: **batched matrix products with a per-row softmax**.

## What this computes & why the GPU helps

Protein language models (PLMs) such as Meta's ESM-2 (650M–15B parameters) learn
evolutionary constraints from hundreds of millions of protein sequences; their
residue embeddings encode structure, function, and mutational effects. ESMFold
uses ESM-2 as a trunk to predict 3D structure without an MSA, far faster than
AlphaFold2 for single sequences. The cost is **multi-head self-attention**: for a
sequence of length `L`, each layer forms an `L×L` attention matrix per head —
`O(L²·d)` work dominated by dense GEMMs (`QKᵀ`, `A·V`) and softmaxes, exactly the
Tensor-Core workloads GPUs excel at. Folding 10M+ UniProt proteins required GPU
clusters precisely because this kernel repeats across ~33 layers × ~20 heads.

**The parallel bottleneck:** the attention rows. Each query residue's attention
distribution over all keys, and the value blend that follows, is independent of
the other queries — so we map **one thread-block to one (head, query-row)** and
let the block's threads cooperate over the `L` keys (logits → shared-memory
softmax → value blend). The output projection is a second, embarrassingly parallel
GEMM (one thread per output element).

## The algorithm in brief

- **Embed:** each residue's amino acid → a `d_model` vector `X[i]` (synthetic
  deterministic embeddings here; trained in a real PLM).
- **Project:** `Q = X·Wq`, `K = X·Wk`, `V = X·Wv`, each split into `H` heads of
  width `d_head = d_model/H`.
- **Attend (per head):** `A = softmax(Q·Kᵀ / √d_head)` over keys (rows sum to 1);
  head output `= A·V`.
- **Combine:** concatenate heads → `Z`; output projection `Y = Z·Wo`.
- **Report:** per-residue output-embedding norm + the residue each one attends to
  most (a "contact"-like readout).

Key algorithms from the catalog: multi-head self-attention (Q×Kᵀ scaling, softmax,
V aggregation); rotary positional embeddings; FlashAttention memory-efficient
attention (all discussed in [THEORY.md](THEORY.md)).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/protein-language-model-inference.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/protein-language-model-inference.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\protein-language-model-inference.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (CMake build)
```

The demo builds if needed, runs on `data/sample/protein_sample.txt`, prints the
per-residue summary, shows the GPU-vs-CPU agreement check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/protein_sample.txt` — a 24-residue
  **synthetic** peptide plus the model shape (`d_model=32, heads=4`). No weights
  are stored: they are generated from an integer hash in `src/attention_math.h`.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print pointers to the real
  trained models (fair-esm / ESM-2, ESMFold) and sequence corpora (UniRef). They
  download nothing — the demo is self-contained on synthetic weights.
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: UniRef50/90 (<https://www.uniprot.org/help/uniref>); ESM
Metagenomic Atlas (<https://esmatlas.com/>); PDB (<https://www.rcsb.org/>);
CATH/SCOP (<https://www.cathdb.info/>).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt): a
per-residue table of output-embedding norms and most-attended residue, ending in
`RESULT: PASS`. The program computes everything on both the **GPU**
(`src/kernels.cu`) and a **CPU reference** (`src/reference_cpu.cpp`) and asserts
they agree — the output embeddings to `max_abs_err ≈ 1.5e-8` and the head-0
attention map to `≈ 7e-9`, both far under the `1e-4` tolerance, with an identical
discrete most-attended-residue readout. A real teaching detail visible in the
output: **identical residues get identical summaries** (every `A` shares a norm,
every `K` shares a norm) because this block has *no positional encoding* — the
motivation for rotary embeddings (THEORY §"Where this sits in the real world").

## Code tour

Read in this order:

1. [`src/attention_math.h`](src/attention_math.h) — the shared `__host__ __device__`
   per-element math (embeddings, projection dot, scaled score, stable softmax).
   Both the CPU and GPU call these, so their arithmetic is identical.
2. [`src/main.cu`](src/main.cu) — loads the protein, runs CPU + GPU, verifies, reports.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the block-per-(head,row) idea.
4. [`src/kernels.cu`](src/kernels.cu) — the three kernels (attention rows, output
   projection, row norms) and the host wrapper.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

- **fair-esm** (<https://github.com/facebookresearch/esm>) — Meta's ESM-2 and
  ESMFold; the reference CUDA inference code and pretrained weights.
- **EvolutionaryScale ESM3** (<https://github.com/evolutionaryscale/esm>) — the
  latest multimodal protein model.
- **ColabFold** (<https://github.com/sokrypton/ColabFold>) — fast MSA + AlphaFold2
  on GPU; contrasts the MSA-based approach ESMFold avoids.
- **FlashAttention** (<https://github.com/Dao-AILab/flash-attention>) — the
  memory-efficient attention kernel production stacks use; the optimization our
  teaching kernel deliberately does *not* do (we materialize the full row).
- **"Attention Is All You Need"** (Vaswani et al. 2017) — the original scaled
  dot-product attention this implements.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Block-per-(head, query-row) attention + GEMM.** A 2-D grid `(L queries × H
heads)`; each block computes one attention row by (1) scoring all `L` keys, (2) a
**shared-memory parallel softmax** (tree max-reduction, then sum-reduction), and
(3) the value blend. A second kernel does the output projection as a dense GEMM
(one thread per output element). This is the same shape FlashAttention and cuBLAS
optimize; we keep it readable rather than fused. Catalog note: production uses
cuDNN / FlashAttention-2 for attention and cuBLAS GEMM for the feed-forward layers,
with Tensor-Core FP16/BF16 mixed precision and dynamic length-bucketed batching.

## Exercises

1. **Add positional encoding.** Implement **rotary positional embeddings** (RoPE)
   on Q and K so identical residues at different positions stop being
   interchangeable — then watch the per-residue norms diverge.
2. **Stack layers.** Wrap the block in a residual + LayerNorm and run `N` blocks;
   confirm CPU/GPU still agree and watch the attention sharpen.
3. **Causal mask.** Add a triangular mask (a residue may only attend to earlier
   ones) and verify the upper triangle of the attention map is zero.
4. **Tile the softmax (FlashAttention idea).** Rewrite `attention_rows_kernel` to
   stream keys in tiles with a running max/sum instead of materializing the whole
   `L`-logit row — the trick that makes long sequences fit in shared memory.
5. **Batch proteins.** Process several sequences at once with length-bucketed
   padding (the real throughput case named in the catalog).

## Limitations & honesty

- **One block, no residual/LayerNorm/MLP, no positional encoding.** A real ESM-2
  stacks ~33 such blocks with rotary embeddings, residual connections, LayerNorm,
  and a feed-forward MLP. We isolate the attention math to teach it cleanly.
- **Synthetic everything.** The sequence is a meaningless synthetic peptide and the
  weights are deterministic hashes, **not** trained parameters — so the embeddings
  carry no biological signal. The point is the *computation*, not a prediction.
- **FP32, full-row softmax.** We materialize the whole attention row (fine at tiny
  `L`); production uses tiled FlashAttention and FP16/BF16 Tensor Cores. The CPU/GPU
  gap (`~1e-8`) comes from a different softmax-sum reduction order; we verify to a
  generous `1e-4` and say so.
- **Not for any clinical or biological use.**
