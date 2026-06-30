# 3.19 — Variant Effect / Pathogenicity Prediction

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Genomics%2C%20Sequencing%20%26%20Bioinformatics-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 3: Genomics, Sequencing & Bioinformatics · Catalog ID `3.19`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

A genetic variant (a single-base substitution, an SNV) flips one DNA letter at one
position. Is it harmless, or does it break a gene? Modern tools answer this by
running the surrounding **sequence context** through a deep neural network twice —
once with the reference allele, once with the alternate — and reporting the
**difference** in the model's output as the predicted effect ("in-silico
mutagenesis"). This project teaches the **GPU inference pattern** behind that idea:
each variant is an independent (reference, alternate) forward-pass pair, so we give
**one GPU thread per variant** and batch-score the whole set at once. To keep the
focus on the CUDA pattern (and to stay honest about scope), the "deep network" here
is a small, fixed-weight 1-D convolutional net standing in for AlphaMissense /
Enformer / a DNA language model — same data-flow, drastically smaller.

## What this computes & why the GPU helps

Predicting whether a DNA variant is pathogenic combines evolutionary conservation
scores (SIFT, PolyPhen), deep mutational scanning models, and increasingly large
genomic foundation models (Nucleotide Transformer, Enformer, AlphaMissense). The
GPU bottleneck is **batched inference of deep networks over millions of variants**:
each SNP generates a pair of (reference, alternate) sequence context windows; the
difference in model output is the predicted effect. AlphaMissense scored all 71 M
possible human missense variants using GPU clusters; Enformer's convolutional-
attention model runs on GPU in batch over 200 kb sequence windows.

**The parallel bottleneck:** the per-variant forward pass. Each variant is scored
**independently** of every other — there are no cross-variant dependencies — so the
problem is *embarrassingly parallel*. Two forward passes (ref + alt) over a length-`L`
context window is hundreds-to-billions of multiply-adds depending on model size, and
you repeat it for every variant in the batch. That is a textbook fit for the GPU's
thousands of cores: in this teaching version each thread scores one variant; in
production each *block* or *stream* handles one network on Tensor Cores. This repo's
toy net runs the **same pattern** at a size you can read end-to-end.

## The algorithm in brief

- **Deep CNN variant-effect scoring** (here a tiny fixed 1-D CNN: conv → ReLU →
  global-max-pool → dense → sigmoid).
- **In-silico mutagenesis / delta score**: `effect = score(ALT window) − score(REF window)`.
- **Log-odds-ratio (LOR) idea**: a per-variant score derived from the *difference*
  of two model outputs (the same shape ESM-1v / Enformer use; we use the raw delta).
- Batched **(ref, alt)** pair inference, one GPU thread per variant, model weights in
  **constant memory** (broadcast to every thread).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation, plus how real tools (AlphaMissense, Enformer, ESM-1v) differ.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/variant-effect-pathogenicity-prediction.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/variant-effect-pathogenicity-prediction.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\variant-effect-pathogenicity-prediction.sln /p:Configuration=Release /p:Platform=x64
```

Both `Debug|x64` and `Release|x64` build with zero warnings. The project links only
`cudart_static.lib` — the CNN forward pass is hand-rolled on purpose (no cuDNN), so
every multiply-add is visible and teachable.

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/variants_sample.txt`, prints the
top-5 most "pathogenic-looking" variants, shows the GPU-vs-CPU agreement check, and
prints a timing line on stderr.

## Data

- **Sample (committed):** `data/sample/variants_sample.txt` — a tiny, **synthetic**
  batch of 12 variants so the demo runs with zero downloads. It is engineered so the
  ranking is interpretable: a few variants whose alternate allele *builds* a planted
  "deleterious" motif rise to the top.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print instructions for
  ClinVar / gnomAD / MaveDB (no credential bypass); `scripts/make_synthetic.py`
  regenerates the sample.
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: ClinVar pathogenic/benign variants
(https://www.ncbi.nlm.nih.gov/clinvar/); gnomAD constraint scores
(https://gnomad.broadinstitute.org/); DMS deep mutational scanning atlas
(https://www.mavedb.org/); HGMD (http://www.hgmd.cf.ac.uk/).

## Expected output

Success looks like `demo/expected_output.txt`:

```
3.19 -- Variant Effect / Pathogenicity Prediction
Batched in-silico mutagenesis: 12 variants, 21-base context, delta = score(ALT) - score(REF)
top-5 most pathogenic-looking variants:
  #1  pos 101096  T>G  delta = +0.321231
  #2  pos 100685  A>G  delta = +0.286602
  #3  pos 100000  T>G  delta = +0.247159
  #4  pos 100274  A>G  delta = +0.142423
  #5  pos 100411  A>C  delta = +0.012040
RESULT: PASS (GPU matches CPU within tol=1.0e-09)
```

The three top hits (`T>G`, `A>G`, `T>G`) are exactly the variants whose alternate
allele completes the planted deleterious 5-mer `CAGCT` at the centre of the window —
a built-in known answer (PATTERNS.md §6). The program computes the per-variant delta
on both the **GPU** (`src/kernels.cu`) and a **CPU reference**
(`src/reference_cpu.cpp`) and asserts they agree within `1e-9`. They actually agree
to `~3e-17` (machine epsilon) because both call the **same** `vep_model.h` code — the
shared `__host__ __device__` core — so the only difference is FMA rounding.

## Code tour

Read in this order:

1. [`src/vep_model.h`](src/vep_model.h) — **start here**: the one shared model and the
   per-variant math (`__host__ __device__`), used verbatim by both CPU and GPU.
2. [`src/main.cu`](src/main.cu) — loads variants, runs CPU + GPU, verifies, ranks, reports.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the thread-mapping idea.
4. [`src/kernels.cu`](src/kernels.cu) — the kernel (one thread per variant) + host wrapper
   (model → constant memory).
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the loader, the fixed-weight model
   initialiser, and the trusted serial baseline.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, CUDA-event timer, host I/O helpers.

## Prior art & further reading

- **AlphaMissense** (https://github.com/google-deepmind/alphamissense) — DeepMind's
  GPU-inferred pathogenicity for all 71 M human missense variants. Study how it frames
  the variant effect as a model-output difference at the protein level.
- **Enformer** (https://github.com/google-deepmind/deepmind-research/tree/master/enformer)
  — dilated-convolution + attention over 200 kb windows; the canonical example of the
  *ref/alt difference = regulatory variant effect* recipe this project miniaturises.
- **EVE / ESM-1v** (https://github.com/facebookresearch/esm) — protein language models
  scoring variants by a **log-odds ratio** of alt vs. ref token probabilities.
- **Nucleotide Transformer** (https://github.com/instadeepai/nucleotide-transformer) —
  a DNA foundation model whose embeddings drive variant-effect predictions.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Independent jobs · constant-memory model** (PATTERNS.md §1, exemplar `1.12`). Each
variant is one independent forward-pass pair → one GPU thread, dispatched with a
grid-stride loop so a fixed grid covers any batch size. The fixed model weights live
in `__constant__` memory: every thread reads the same weights and none writes them, so
the constant cache **broadcasts** one address to a whole warp per transaction — the
same trick `1.12` uses for its query fingerprint. Production tooling listed in the
catalog (cuDNN, TensorRT, Tensor-Core BF16, CUDA Graphs) is the heavy-duty version of
this same batched-inference pattern; THEORY.md "Where this sits in the real world"
explains the upgrade path.

## Exercises

1. **Bigger batch, real speed-up.** Regenerate a large sample
   (`python scripts/make_synthetic.py --n 2000000`) and compare CPU vs. GPU timing on
   stderr. At what `n` does the GPU overtake the CPU? (See the honest-timing note in
   THEORY.md.)
2. **Saturation mutagenesis.** Extend `main.cu` so that for one chosen position it
   scores **all three** alternate alleles and prints the full per-base effect — the
   "in-silico saturation mutagenesis" tools actually run.
3. **A real log-odds ratio.** Replace the raw delta with `log(p_alt/(1−p_alt)) −
   log(p_ref/(1−p_ref))` (the logit difference) and confirm it re-ranks sensibly.
4. **Shared-memory model.** The model already fits in constant memory; try staging the
   windows into shared memory for a block and measure whether it helps (hint: per-thread
   work here is compute-bound, so probably not — explain why).
5. **Second precision.** Add an FP32 build path of `vep_score_window` and measure the
   CPU/GPU divergence vs. the FP64 path. Which tolerance would you then document?

## Limitations & honesty

- **The model is synthetic and untrained.** `init_model()` writes fixed, hand-designed
  weights with two planted motifs. It is **not** trained on ClinVar/gnomAD/DMS and
  produces **no clinically meaningful** scores. It exists to teach the GPU batched-
  inference pattern, not genetics (CLAUDE.md §8).
- **The data is synthetic** and labelled as such everywhere. The "pos" coordinates are
  invented; the windows are random background with motifs planted to make the demo
  legible.
- **Reduced scope vs. the catalog.** Real variant-effect predictors are CNNs/transformers
  with millions–billions of parameters run via cuDNN/TensorRT on Tensor Cores; this is a
  ~few-KB fixed CNN run by hand. THEORY.md "Where this sits in the real world" spells out
  exactly what changes at production scale.
- **No diagnostic claims.** Nothing here may inform a real medical decision.
