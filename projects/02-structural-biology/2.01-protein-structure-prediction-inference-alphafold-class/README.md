# 2.1 — Protein Structure Prediction Inference (AlphaFold-class)

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Structural%20Biology%20%26%20Protein%20Science-lightgrey) ![scope](https://img.shields.io/badge/scope-reduced--teaching%20version-orange)

> **🟢 Beginner · Established** — Domain 2: Structural Biology & Protein Science · Catalog ID `2.1`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

AlphaFold and its successors (ESMFold, OpenFold, RoseTTAFold, Boltz-1) predict a
protein's 3-D structure from its amino-acid sequence using deep learning. At the
heart of every one of them is a transformer **self-attention** layer: a step that
lets each residue "look at" every other residue and pull in context. This project
is a **reduced-scope teaching version** that implements exactly that one building
block on the GPU — **scaled dot-product self-attention** over a protein's residue
representations — with a heavily-commented CUDA kernel, a CPU reference that
computes the identical math, and a demo that verifies they agree. You will not
fold a protein here; you will learn, end to end and with no black boxes, the
single most important and most GPU-hungry operation inside the models that do.

## What this computes & why the GPU helps

A protein of `L` residues is represented by an `L × d` matrix of feature vectors.
A self-attention head projects it into three matrices — **Q**uery, **K**ey,
**V**alue (each `L × d`) — and computes, for every residue `i`:

```
Out[i] = Σ_j  softmax_j( (Q[i]·K[j]) / sqrt(d) ) · V[j]
```

i.e. residue `i`'s new vector is a softmax-weighted average of all residues'
value vectors, weighted by how similar `i`'s query is to each key.

**The parallel bottleneck.** Computing all `L²` pairwise affinities `Q[i]·K[j]`
costs `O(L² · d)`, and AlphaFold stacks **dozens** of attention layers across
**recycling iterations** over MSAs with hundreds of rows — which is why a single
500-residue prediction takes minutes on a GPU but ~12 hours on a CPU. The work is
massively parallel: the `L` output rows are independent, and within a row all `L`
scores are independent. We exploit both. This project teaches the schoolbook GPU
attention kernel; production systems use **FlashAttention** (the same math, fused
and tiled to avoid materializing the `L × L` score matrix).

## The algorithm in brief

- **Project** the residue representation into Q, K, V (`L × d` each). _(Here the
  projections are supplied directly as synthetic input; in a real model they are
  learned linear layers.)_
- **Score:** `S[i][j] = (Q[i]·K[j]) / sqrt(d)` — scaled dot products.
- **Softmax** each row of `S` (numerically stable: subtract the row max first).
- **Mix:** `Out[i] = Σ_j softmax(S[i])[j] · V[j]` — weighted sum of value vectors.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation, including how this single head sits inside the Evoformer.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/protein-structure-prediction-inference-alphafold-class.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/protein-structure-prediction-inference-alphafold-class.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\protein-structure-prediction-inference-alphafold-class.sln /p:Configuration=Release /p:Platform=x64
```

This project links only the CUDA runtime (`cudart_static.lib`) — the attention
kernel is hand-rolled, on purpose, so nothing is a black box.

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if the CMake build is used)
```

The demo builds if needed, runs on `data/sample/`, prints the per-residue
attention result, shows the GPU-vs-CPU agreement check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/attention_sample.txt` — a tiny,
  **synthetic** set of Q/K/V matrices (`L=6`, `d=32`) so the demo runs with zero
  downloads. It is engineered so each residue attends to itself (a verifiable
  known answer).
- **Larger synthetic:** `python scripts/make_synthetic.py --L 64`.
- **Provenance & license:** see [data/README.md](data/README.md). The sample is
  synthetic (CC0); no real structures, MSAs, or model weights are shipped.

Real-world databases (for further study, not needed here): AlphaFold DB
(<https://alphafold.ebi.ac.uk/>), RCSB PDB (<https://www.rcsb.org>), UniProt
(<https://www.uniprot.org>), CAMEO/CASP15 (<https://www.cameo3d.org>).
`scripts/download_data.ps1` / `.sh` print these pointers without downloading
gated/large assets.

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt): six
lines reporting which residue each query attends to most (every residue → itself)
and the output-row norms, ending in `RESULT: PASS`. The program computes the
attention output on both the **GPU** (`src/kernels.cu`) and a **CPU reference**
(`src/reference_cpu.cpp`) and asserts they agree within `1e-5` — that agreement is
the correctness guarantee. The actual `max_abs_err` (~`4.8e-7`) is printed on
stderr.

## Code tour

Read in this order:

1. [`src/attention_core.h`](src/attention_core.h) — the shared `__host__ __device__`
   math (dot product, the `1/sqrt(d)` scale, the stable-softmax exponential). Both
   CPU and GPU call these, so they are numeric twins.
2. [`src/main.cu`](src/main.cu) — loads data, runs CPU + GPU, verifies, reports.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the "one block per
   output row, cooperate via shared memory" idea.
4. [`src/kernels.cu`](src/kernels.cu) — the three-phase attention kernel (score →
   softmax reduction → weighted value sum) and the host wrapper.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline
   and the data loader.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

- **AlphaFold2** (<https://github.com/google-deepmind/alphafold>) — the official
  DeepMind implementation; study `modules.py`/`modules_multimer.py` to see the
  Evoformer's row/column attention and triangle updates in real code.
- **OpenFold** (<https://github.com/aqlaboratory/openfold>) — a trainable,
  GPU-friendly PyTorch reimplementation of AF2; the clearest place to read the
  attention and IPA modules.
- **ESMFold / ESM** (<https://github.com/facebookresearch/esm>) — MSA-free
  prediction from a protein language model; shows attention used as the *only*
  evolutionary signal.
- **Boltz-1** (<https://github.com/jwohlwend/boltz>) — a fully open AF3-level
  model with diffusion-based structure generation.
- **FlashAttention** (Dao et al. 2022) — the fused, memory-efficient attention
  kernel that production systems actually run; our kernel is its un-fused ancestor.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## Exercises

1. **Multi-head attention.** Split the `d=32` channels into `h` heads of width
   `d/h`, run attention independently per head, and concatenate. How does the
   per-block shared-memory budget change?
2. **Causal / banded masking.** Add a mask so residue `i` may only attend to
   residues within a window — a common trick for very long sequences. Where in
   the kernel does the mask go, and why before the softmax?
3. **The FlashAttention idea.** Modify the kernel to process keys/values in
   **tiles**, keeping a running max and running denominator (the "online softmax")
   so the full `L × L` score matrix is never stored. Verify it still matches the
   CPU. This is the single most important attention optimization.
4. **FP16/BF16 inputs.** Store Q/K/V in `__half` (as real models do) but
   accumulate scores in FP32. Measure the accuracy hit against the FP32 reference.
5. **Bigger problems.** Sweep `L` from 6 to 4096 (`make_synthetic.py --L`) and plot
   CPU vs GPU time. At what `L` does the GPU overtake the CPU, and why?

## Limitations & honesty

- **This is one attention head, not AlphaFold.** A real prediction adds: learned
  Q/K/V/output projections, **multi-head** attention, **MSA row *and* column**
  attention, **triangle multiplicative updates** and **triangle attention** on a
  pair representation, the **Structure Module** with **Invariant Point Attention
  (IPA)** that turns representations into 3-D coordinates, **recycling**, and
  confidence heads (**pLDDT**, **PAE**). Those are described in
  [THEORY.md §7](THEORY.md#7-where-this-sits-in-the-real-world) but **not**
  implemented here.
- **The data is synthetic** and labeled synthetic everywhere. The Q/K/V matrices
  are a hand-built toy with an embedded known answer; they are not a real protein,
  MSA, or learned weights.
- **No structure is predicted and no clinical claim is made.** The output is a
  context-mixed feature matrix, an intermediate quantity — useful for *learning*
  the attention mechanism, nothing more.
- **Precision/determinism.** Scores and accumulations are done in `double` and the
  weighted value sum is accumulated in the same residue order on CPU and GPU, so
  results match to ~`1e-7`; we verify to a documented `1e-5`. See
  [THEORY.md §5](THEORY.md#5-numerical-considerations).
