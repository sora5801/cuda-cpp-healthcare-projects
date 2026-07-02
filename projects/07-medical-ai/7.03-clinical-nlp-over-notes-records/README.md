# 7.3 — Clinical NLP over Notes & Records

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Medical%20AI%20%26%20Clinical%20Deep%20Learning-lightgrey)

> **🟢 Beginner · Established** — Domain 7: Medical AI & Clinical Deep Learning · Catalog ID `7.3`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

Clinical NLP models read free-text electronic health records — discharge
summaries, radiology reports, nursing notes — with **transformer** language
models (BERT-style) to do named-entity recognition, coreference/relation
extraction, ICD coding, and phenotyping. The computational heart of every such
model is **multi-head self-attention**: each token in a note computes a weighted
average of the other tokens, where the weights come from scaled dot products of
learned *query* and *key* vectors. This project implements **one self-attention
encoder block** end to end — projections, scaled dot-product attention, masking,
softmax, and the context mixing — on the GPU with **cuBLAS batched matrix
multiply** for the GEMM-dominated steps and a hand-written softmax kernel, and
verifies it against a plain-C++ reference. It is a **reduced-scope teaching
version** (CLAUDE.md §13): it computes the *mechanism* a clinical transformer runs
in each layer, using deterministic fabricated weights rather than a trained model,
on a tiny synthetic note batch that plants a coreference-like link for the demo to
recover.

## What this computes & why the GPU helps

Applies transformer language models to de-identified EHR free-text for NER,
relation extraction, ICD coding, phenotyping, and clinical event prediction.
BERT-style pretraining on billions of clinical tokens is highly GPU-bound:
multi-head self-attention scales **O(n²)** in sequence length, making long
clinical notes expensive; Flash Attention reduces the *memory* cost to near-linear
and enables 8192-token contexts. The parallel bottleneck is the **batched matrix
multiplications** in each transformer layer, which exploit GPU tensor cores.

**The parallel bottleneck (what we accelerate here):** the two matmuls inside
attention — `scores = Q·Kᵀ` and `context = softmax(scores)·V` — done independently
for every (note, head) pair. For `B` notes, `H` heads, sequence length `S`, and
head width `dh`, that is `B·H` independent `S×dh · dh×S` and `S×S · S×dh` GEMMs.
We issue them as **cuBLAS strided-batched DGEMM** (one launch per note over its
heads); on real hardware these dispatch to tensor cores. The softmax between them
is the one non-GEMM step and is a hand-written reduction kernel.

## The algorithm in brief

- **Embedding lookup**: gather each token id's `D`-dim embedding into `X` `[S×D]`.
- **Linear projections**: `Q = X·Wq`, `K = X·Wk`, `V = X·Wv` (dense GEMMs).
- **Multi-head split**: partition the `D` columns into `H` heads of width `dh = D/H`.
- **Scaled dot-product attention** per head: `scores = Q_h·K_hᵀ / √dh`.
- **Padding mask**: `[PAD]` key positions get score `−∞` so they receive no weight.
- **Softmax** (numerically stable, max-subtracted) turns each score row into a
  probability distribution over keys.
- **Context mixing**: `O_h = A_h·V_h`; concatenate heads back to `O` `[S×D]`.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation, including why attention is O(n²) and what Flash Attention changes.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)). This project links
**cuBLAS** (`cublas.lib`) for the attention GEMMs — already wired into the
`.vcxproj` (both configs) and `CMakeLists.txt`.

1. Open `build/clinical-nlp-over-notes-records.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/clinical-nlp-over-notes-records.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\clinical-nlp-over-notes-records.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (uses the optional CMake build)
```

The demo builds if needed, runs on `data/sample/notes_sample.txt`, prints the
result, shows the GPU-vs-CPU agreement check, and prints a per-stage timing line.

## Data

- **Sample (committed):** `data/sample/notes_sample.txt` — a tiny, **synthetic**
  batch of 4 tokenized "notes" over a 15-token toy vocabulary + a fabricated
  embedding table, so the demo runs offline with zero downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print how to obtain the
  real corpora legally (they are all credentialed).
- **Provenance & license:** see [data/README.md](data/README.md).

The real clinical corpora are **credentialed** and cannot be redistributed:
MIMIC-IV Clinical Notes (<https://physionet.org/content/mimic-iv-note/>), the
i2b2/n2c2 NLP challenges (<https://n2c2.dbmi.hms.harvard.edu/>), MTSamples
(<https://mtsamples.com/>), and MedQA/MedMCQA. The committed sample is therefore
synthetic and labeled synthetic everywhere.

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt). The
program computes the attention block on both the **GPU** (`src/kernels.cu`) and a
**CPU reference** (`src/reference_cpu.cpp`) and asserts the attention weights
`[B·H·S·S]` and the output embeddings `[B·S·D]` agree entrywise within a
documented tolerance (`1e-11`; they match to ~`1e-16` in practice). The headline,
human-meaningful result is the recovered coreference link: the pronoun `he`
attends most strongly to `patient` in all 4 notes.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads the note batch, runs CPU + GPU, verifies, reports.
2. [`src/attn_core.h`](src/attn_core.h) — the shared `__host__ __device__` math:
   projection-weight recipe, stable softmax helpers, attention entropy.
3. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline
   (loader + the whole attention block in obvious loops).
4. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + how the block splits
   into GEMMs, a batched GEMM, and a softmax kernel.
5. [`src/kernels.cu`](src/kernels.cu) — the device kernels and the cuBLAS
   (batched) DGEMM calls, with the row-major↔column-major reasoning at each site.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, CUDA-event timer, I/O helpers.

## Prior art & further reading

- **BioClinicalBERT** (<https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT>) —
  BERT pretrained on MIMIC-III notes; the canonical clinical encoder. Study its
  tokenizer and the `[CLS]`-pooling convention we mimic.
- **Clinical ModernBERT** (<https://github.com/Simonlee711/Clinical_ModernBERT>) —
  ModernBERT on 13B tokens of PubMed + MIMIC-IV with 8192-token context; shows how
  RoPE + Flash Attention extend the context window this project's O(n²) block
  cannot.
- **medSpaCy** (<https://github.com/medspacy/medspacy>) — a rule+model clinical NLP
  pipeline; good for seeing NER/section-detection end to end.
- **GatorTron** (<https://huggingface.co/UFNLP/gatortron-base>) — a large clinical
  LLM (82B tokens); illustrates the data-parallel, multi-GPU pretraining the
  catalog names (verify URL).

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Dense linear algebra → cuBLAS**, plus a **block-per-row reduction** for softmax
(PATTERNS.md §1). The three projections are single DGEMMs; the two attention
matmuls are `cublasDgemmStridedBatched` over the `B·H` (note, head) pairs; the
softmax is a hand-written kernel (one block per attention row, cooperative
max + sum reductions in shared memory). The shared `__host__ __device__` core
(PATTERNS.md §2) guarantees the CPU reference and GPU kernels run identical
per-element math. Real systems fuse the score-GEMM, softmax, and context-GEMM into
a single kernel that never materializes the `S×S` scores — that is **Flash
Attention** (THEORY §"real world").

## Exercises

1. **Add a causal mask.** Change the mask so token `i` may only attend to
   positions `≤ i` (a decoder / autoregressive mask). Confirm the upper triangle
   of every attention matrix becomes zero.
2. **Sweep the head count.** Regenerate the sample with `--dim 16 --heads 4` and
   `--heads 1`; observe how splitting into more heads changes the recovered
   attention pattern and the per-head entropy.
3. **Sharpen the signal.** Modify the embedding recipe in `make_synthetic.py` so
   `he` and `patient` align more strongly (e.g. scale their shared dims up), and
   watch the `he→patient` weight climb well above uniform.
4. **Fuse for Flash Attention.** Replace the three-stage GPU pipeline with a single
   kernel that, per query row, streams over key blocks keeping a running max and
   running sum (the online-softmax trick) so the `S×S` matrix is never stored.
   Verify it still matches the CPU reference.
5. **FP32 vs FP64.** Switch the pipeline to `float`/`cublasSgemm` and measure how
   much the CPU-vs-GPU tolerance must loosen — a concrete lesson in transformer
   numerics.

## Limitations & honesty

- **Reduced-scope teaching version.** This implements the **attention mechanism**
  of a clinical transformer, not a trained model. There is no pretraining, no
  fine-tuning, no tokenizer, no feed-forward sublayer, no residual/LayerNorm, and
  no task head (NER/ICD/RE). Those are described in [THEORY.md](THEORY.md).
- **Fabricated weights.** The embeddings and the `Wq/Wk/Wv` projections come from
  fixed integer recipes, chosen for determinism and legibility — **not learned**.
  Consequently the attention is fairly diffuse (weights only modestly above
  uniform); the *argmax* recovers the planted link, which is the teachable point.
- **Synthetic data.** The note batch is invented and labeled synthetic
  everywhere; it contains no real patient text and implies **no clinical
  validity**. Nothing here is a diagnosis, prognosis, or clinical recommendation.
- **Tiny problem, honest timing.** At `S=8` the GPU is launch/copy-bound and slower
  than the CPU; the timing is a teaching artifact, not a benchmark. Attention's
  O(S²) cost — and the GPU's advantage — only bite at real sequence lengths.
- **No RoPE / no Flash Attention.** Positional information and the memory-efficient
  fused kernel the catalog names are left as exercises / described in THEORY.
