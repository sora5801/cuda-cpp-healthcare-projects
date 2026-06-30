# Push 2026-06-30 #07 -- phase2 batch4e domain3-complete

> Push-note (CLAUDE.md section 7.1). Fifth domain-3 batch — and a **milestone**: the last 5
> projects bring **Domain 3 (Genomics, Sequencing & Bioinformatics) to 30/30**.

## 1. Summary

Five projects complete **Domain 3 — Genomics, Sequencing & Bioinformatics, now 30/30** — the
**third of 14 domains** finished end-to-end — and take the collection to **106 -> 111 / 301
(36.9%)**. This batch is the "classic bioinformatics data structures" cluster: GPU BAM sorting
& deduplication, suffix array / BWT / FM-index construction, profile-HMM Viterbi/Forward,
motif finding, and pangenome graph construction. It introduces a **third CUDA library to the
collection — Thrust** (3.26 `sort_by_key` + `reduce_by_key`), now documented in BUILD_GUIDE
§7c. Each was built in its own folder by one worker and re-verified by the lead.

## 2. What changed

Five new fully-implemented projects under `projects/03-genomics/`:

- [`3.26` GPU BAM Sorting & Deduplication](../projects/03-genomics/3.26-gpu-bam-sorting-deduplication)
- [`3.27` Suffix Array / BWT / FM-Index Construction](../projects/03-genomics/3.27-suffix-array-bwt-fm-index-construction)
- [`3.28` Profile HMM (Viterbi / Forward)](../projects/03-genomics/3.28-profile-hmm-viterbi-forward)
- [`3.29` Motif Finding in Genomic Sequences](../projects/03-genomics/3.29-motif-finding-in-genomic-sequences)
- [`3.30` Pangenome Graph Construction](../projects/03-genomics/3.30-pangenome-graph-construction)

Plus a `docs/BUILD_GUIDE.md` §7c (how to build Thrust/CUB projects). `docs/STATUS.md` -> 5
marked **done** (111/301; **domain 3 = 30/30**). `CHANGELOG.md` indexed.

## 3. New projects (didactic blurb each)

- **3.26 BAM Sort & Dedup** — **Thrust** `sort_by_key` radix coordinate sort + segmented
  `reduce_by_key` "keep best per fragment" dedup, shared integer key/compare core -> exact
  CPU==GPU (358 planted duplicates recovered). **First Thrust project**; the BUILD_GUIDE §7c
  note captures the `.vcxproj` flags it needed (`/Zc:preprocessor`, `-std=c++17`, Debug
  `-diag-suppress`).
- **3.27 Suffix Array / BWT / FM-Index** — GPU **prefix-doubling** SA construction via a
  hand-rolled deterministic LSD radix sort (no external lib), then BWT derivation and FM-index
  **backward search** (shared key-packing core -> exact CPU==GPU). Recovers the planted motif's
  6 hits. Two algorithmic bugs were found *because* the exact-match verification flagged them.
- **3.28 Profile HMM** — a profile-HMM database search with per-column M/I/D states, one thread
  per sequence (profile in constant memory): log-space **Viterbi** (max-sum) and **Forward**
  (log-sum-exp). Shared `phmm.h` -> bit-exact CPU==GPU; the planted homolog ranks #1 by ~54 nats.
  The HMMER `hmmsearch` core.
- **3.29 Motif Finding** — **MEME OOPS Expectation-Maximization**: the expensive E-step (score
  every length-W window against a PWM log-odds table) runs as a kernel (one thread per window,
  table in constant memory). Shared `window_score()` -> bit-exact; recovers the TGACGTCA core
  (~8.27 bits).
- **3.30 Pangenome Graph Construction** — reduced ODGI-style **1-D graph layout** via
  **SMACOF / stress majorization** (one thread per stress term with atomic scatter + one thread
  per node to apply), fixed-point integer atomics -> bit-exact CPU==GPU. Recovers the correct
  node order (variant nodes beside their neighbours); stress drops ~100x.

All five are clearly-labeled **reduced-scope teaching versions** (synthetic data, labeled
synthetic), with production tools (samtools/Picard, BWA/bowtie2 indexers, HMMER, MEME Suite,
ODGI/vg) named in each `THEORY.md`.

## 4. How to build & run

```powershell
cd projects/03-genomics/3.27-suffix-array-bwt-fm-index-construction   # (or any)
msbuild build/*.sln /p:Configuration=Release /p:Platform=x64
./demo/run_demo.ps1      # -> RESULT: PASS (GPU matches CPU)
```

3.26 uses **Thrust** (header-only, no extra `.lib`; see BUILD_GUIDE §7c for the `.vcxproj`
flags). The others are pure custom kernels.

## 5. What to study here

Domain 3 is now a **complete worked tour** of GPU genomics. Across the domain: the
Smith-Waterman family (flagship 3.01 + banded 3.21, spliced 3.23, event-align 3.24, graph 3.13),
DP beyond alignment (3.09 Felsenstein, 3.10 Nussinov, 3.28 HMM), data structures (3.06 hash
tables, 3.27 SA/BWT/FM-index, 3.26 Thrust sort), ML inference (3.04 CTC, 3.18 attention, 3.19
CNN), and library pipelines (3.11 cuBLAS). Exercise: read 3.01, 3.13, 3.21, 3.23, 3.24 back to
back — five specializations of one recurrence.

## 6. Verification (lead-independent, not self-reports)

- ✅ **Boundaries:** only the 5 project folders + `docs/BUILD_GUIDE.md` changed; no artifacts.
- ✅ **Clean rebuild** (`/t:Rebuild`, fat arch list) of all 5 in both `Release|x64` and
  `Debug|x64`: **EXIT=0, 0 warnings, 0 errors** (10/10 builds), incl. the Thrust project 3.26.
- ✅ All 5 **demos PASS**: GPU==CPU (BAM/SA/motif exact; HMM 1e-4; pangenome layout matches).
- ✅ `verify_project.py` -> **DONE** for all 5 (comment ratios **0.94–1.06**).
- ✅ **Domain-3 sweep:** 30/30 markers `done`.
- **Workflow:** 5 agents, ~0.96M agent tokens, 410 tool uses.
- **Environment:** RTX 2080 (SUPER), `sm_75`, CUDA 13.3, VS 2026 (`v145`).

## 7. Known limitations / TODOs

- All five are **reduced-scope teaching versions**: a small BAM, short strings for SA/BWT, a
  small profile HMM, a synthetic motif set, a tiny pangenome. Labeled synthetic; production
  scale described in each THEORY.md.

## 8. Next push preview

**Domain 4 — Medical Imaging & Image Reconstruction (33 projects).** Flagship `4.01` (CT
filtered backprojection / FDK) is already done; the build-out continues easiest-first through
the remaining 32 in ~6-project batches. Three of 14 domains complete (100 + 14 flagships =
111 projects); 11 domains (190 projects) to go. Same workflow, lead-verified, one push-note
per batch.
