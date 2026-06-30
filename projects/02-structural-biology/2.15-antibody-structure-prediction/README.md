# 2.15 — Antibody Structure Prediction

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Structural%20Biology%20%26%20Protein%20Science-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 2: Structural Biology & Protein Science · Catalog ID `2.15`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._
>
> **⚠️ Reduced-scope teaching version (CLAUDE.md §13).** Full antibody *structure*
> prediction (IgFold, ABodyBuilder3) is an attention-based deep-learning pipeline
> with hundreds of MB of trained weights — not a single didactic CUDA kernel. This
> project teaches the load-bearing biology and the GPU pattern that sit **under**
> high-throughput antibody work: **library screening by CDR similarity.** It does
> **not** predict 3-D coordinates. See [THEORY.md](THEORY.md) for the full pipeline
> and exactly where this piece fits.

## Summary

Given one **query** antibody (described by its six hypervariable **CDR loops**)
and a **library** of antibodies, rank the library by how similar each member's
CDRs are to the query's — weighting **CDR-H3** most, because that loop dominates
antigen specificity. Each CDR pair is scored with a **BLOSUM62** substitution
matrix; the antibody-level score is a CDR-weighted sum. Every query-vs-library
comparison is independent, so we put **one GPU thread on each library antibody** —
the same "score one query vs N items" pattern as project 1.12 (Tanimoto search),
and the screening step that real tools accelerate to "thousands of sequences per
GPU-hour."

## What this computes & why the GPU helps

In the wild, antibody structure prediction is specialized because the **CDR-H3
loop is hypervariable and controls antigen specificity**. Tools like IgFold,
ABodyBuilder3, and IMGT-optimized AlphaFold2 models predict full Fv-region
structures including the flexible CDR loops, and GPU inference enables
high-throughput prediction for **antibody library screening**. This project
isolates the *screening / similarity-ranking* sub-problem (not the folding):
score one query's CDRs against a large library and return the closest matches.

**The parallel bottleneck:** the per-pair CDR scoring — a substitution-matrix sum
over the six CDR fields, `O(N · 144)` integer operations for `N` library
antibodies — dominates a library screen, and every pair is independent. We
parallelize it across the library dimension `N` (one thread per antibody), keep
the shared query in **constant memory** (broadcast to every thread), and stream
the library from global memory.

## The algorithm in brief

- **CDR loops:** six hypervariable loops per antibody Fv (heavy H1/H2/H3, light
  L1/L2/L3); CDR-H3 is the longest and most variable.
- **Per-CDR score:** ungapped BLOSUM62 column-sum of two equal-length, padded CDR
  fields: `score(a,b) = Σ_i S[a_i][b_i]`.
- **Antibody score:** CDR-weighted sum of the six per-CDR scores, **CDR-H3 ×3**.
- **Screen:** score the query against all `N` library antibodies, then take the
  **top-K** highest scores (the hits).

See [THEORY.md](THEORY.md) for the science → math → algorithm → GPU-mapping derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/antibody-structure-prediction.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/antibody-structure-prediction.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\antibody-structure-prediction.sln /p:Configuration=Release /p:Platform=x64
```

Only the CUDA runtime (`cudart_static.lib`) is linked — no extra CUDA libraries.

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/antibodies_sample.txt`, prints the
**top-5 hits** with a per-hit CDR-H3 breakdown, shows the GPU-vs-CPU agreement
check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/antibodies_sample.txt` — 1 query + 24
  **synthetic** antibodies, each with six CDR loops. The generator plants a known
  answer: `mAb_07` is a near-copy of the query, `mAb_18` shares the query's CDR-H3.
- **Full dataset:** real antibody CDRs from **SAbDab / Thera-SAbDab / OAS** — see
  `scripts/download_data.ps1` / `.sh` (they print URLs and a conversion recipe; no
  download or credential bypass).
- **Bigger synthetic set:** `python scripts/make_synthetic.py --n 1048576`.
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: SAbDab — Structural Antibody Database (https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/); OAS (Observed Antibody Space) — 2B antibody sequences (https://opig.stats.ox.ac.uk/webapps/oas/); CASP-Ab benchmarks; Thera-SAbDab — therapeutic antibody database (https://opig.stats.ox.ac.uk/webapps/newsabdab/therasabdab/).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt). The
program computes all `N` scores on the **GPU** (`src/kernels.cu`) and on a **CPU
reference** (`src/reference_cpu.cpp`) and asserts they agree. Because the scoring
is **exact integer arithmetic** shared by both paths (the `__host__ __device__`
core in `src/antibody.h`), the two agree **bit-for-bit** (`max_abs_err = 0`). The
top hits are `mAb_07` (516) and `mAb_18` (373), recovering the planted answer.

## Code tour

Read in this order:

1. [`src/antibody.h`](src/antibody.h) — the shared `__host__ __device__` scoring
   core: amino-acid encoding, the BLOSUM62 matrix, per-CDR and antibody-level scores.
2. [`src/main.cu`](src/main.cu) — loads data, runs CPU + GPU, verifies, prints top-K.
3. [`src/reference_cpu.h`](src/reference_cpu.h) — the data model (`AntibodyLibrary`) + loader/reference prototypes.
4. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the constant-memory / one-thread-per-antibody idea.
5. [`src/kernels.cu`](src/kernels.cu) — the kernel (grid-stride, constant-memory query) + host wrapper.
6. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline + the text loader.
7. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

- **IgFold** (<https://github.com/Graylab/IgFold>) — fast antibody structure prediction on GPU; study its CDR-H3 handling and IMGT numbering.
- **ABodyBuilder3** (<https://github.com/brennanaba/ABodyBuilder3>) — GPU-optimized AF2-style antibody model using ESM-2 embeddings and OpenFold vectorization; the production analogue of the *full* problem.
- **ANARCI** (<https://github.com/oxpig/ANARCI>) — IMGT/Chothia/Kabat antibody numbering; how you delimit the six CDRs we score here.
- **AbDiffuser** / **AbNatiV** (OPIG) — antibody sequence+structure diffusion and naturalness scoring; the design/developability frontier.

Study these for the production approach; reimplement didactically rather than copying (CLAUDE.md §2).

## CUDA pattern used here

Constant memory (broadcast query record) · one thread per library antibody ·
grid-stride loop over the library · a shared `__host__ __device__` scoring core so
CPU and GPU run identical integer math · independent outputs (no atomics/shared
mem) · top-K on the host. This is the "score one query vs N items, each
independent" pattern (PATTERNS.md §1; exemplar 1.12). The *full* pipeline instead
uses cuDNN/Flash-attention transformers for the ESM-2 backbone and a structure
module — out of scope here (see THEORY).

## Exercises

1. **Gapped CDR scoring.** Replace the ungapped column-sum with a
   **Needleman-Wunsch** global alignment per CDR (so CDRs of different lengths
   align properly). Reuse the wavefront idea from project 3.01. Does CDR-H3's rank
   change?
2. **Tune the CDR weights.** The ×3 CDR-H3 weight is a teaching choice. Make the
   six weights command-line arguments and watch how the ranking shifts; which
   weighting best separates `mAb_07`/`mAb_18` from the noise?
3. **Top-K on the GPU.** Move the host `partial_sort` onto the device with
   `cub::DeviceRadixSort` or a Thrust `sort_by_key`. Does it help at `n = 24`? At
   `n = 1,000,000` (`make_synthetic.py --n 1000000`)?
4. **Warp-per-antibody.** Assign one *warp* to each antibody and reduce the 144
   column scores with `__shfl_down_sync`. When does that beat one thread per antibody?
5. **Real CDRs.** Number a handful of real Fv sequences with ANARCI, convert to
   the loader format, and screen a therapeutic antibody against the rest.

## Limitations & honesty

- **This is screening, not structure prediction.** There are **no 3-D
  coordinates** anywhere in this project — it ranks antibodies by CDR *sequence*
  similarity. The README header and THEORY say exactly what the full IgFold/
  ABodyBuilder3 pipeline does that this omits.
- The sample is **synthetic**; the sequences are invented and carry **no
  biological meaning**. The "hits" are planted so the demo is interpretable.
- Scoring is **ungapped** on length-matched, padded CDR fields — the simplest
  correct teaching version. Production tools align gapped CDRs and model 3-D loop
  geometry, disulfides, and framework context.
- The CDR weights (CDR-H3 ×3, others ×1) are illustrative, not a calibrated model.
- Timing is a **teaching artifact**: at `n = 24` the GPU is dominated by launch/
  copy overhead; its advantage appears at library scale (millions of antibodies).
