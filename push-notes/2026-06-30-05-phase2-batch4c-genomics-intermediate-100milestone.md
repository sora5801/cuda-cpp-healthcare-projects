# Push 2026-06-30 #05 -- phase2 batch4c genomics intermediate (100-project milestone)

> Push-note (CLAUDE.md section 7.1). Third domain-3 batch, 6 Intermediate genomics projects —
> and a round-number **milestone: 100 / 301 projects complete (one-third)**.

## 1. Summary

Six more **domain-3 (genomics) Intermediate** projects are complete, taking the collection
to **94 -> 100 / 301 — one-third of the catalog** — with domain 3 now at **19/30**. This
batch is the "modern sequence-analysis" cluster: single-cell RNA-seq, pangenome graph
alignment, metagenomic classification, Hi-C 3D-genome analysis, sequence error correction,
and CRISPR guide design. It also generalizes the alignment DP one step further — **3.13 runs
Smith-Waterman over a sequence DAG** (a pangenome graph), not just two strings. Each was
built in its own folder by one worker and re-verified by the lead. A small docs touch-up
(`PATTERNS.md` cuBLAS/cuSOLVER exemplars) rides along.

## 2. What changed

Six new fully-implemented projects under `projects/03-genomics/`:

- [`3.12` Single-Cell RNA-seq Analysis](../projects/03-genomics/3.12-single-cell-rna-seq-analysis)
- [`3.13` Pangenome Graph Alignment](../projects/03-genomics/3.13-pangenome-graph-alignment)
- [`3.14` Metagenomic Taxonomic Classification](../projects/03-genomics/3.14-metagenomic-taxonomic-classification)
- [`3.15` Hi-C / 3D Genome Contact Analysis](../projects/03-genomics/3.15-hi-c-3d-genome-contact-analysis)
- [`3.16` Sequence Error Correction](../projects/03-genomics/3.16-sequence-error-correction)
- [`3.17` CRISPR Guide Design & Off-Target Scoring](../projects/03-genomics/3.17-crispr-guide-design-off-target-scoring)

Plus a `docs/PATTERNS.md` update (§5 now lists `3.11` cuBLAS `Dgemm`, `1.08` batched cuSOLVER,
`2.20` cuSOLVER-PCA as library exemplars). `docs/STATUS.md` -> 6 marked **done** (100/301).
`CHANGELOG.md` indexed.

## 3. New projects (didactic blurb each)

- **3.12 Single-Cell RNA-seq** — library-size normalize (counts-per-target + log1p) + an
  exact brute-force **KNN cell graph** (shared HD math core, neighbour indices match CPU
  exactly). Synthetic 3-type data recovers 100% KNN label purity. The first stage of every
  scRNA-seq pipeline (before Leiden/UMAP).
- **3.13 Pangenome Graph Alignment** — **Smith-Waterman generalized over a sequence DAG**:
  a per-node anti-diagonal wavefront kernel in topological order, with the cross-node max in
  a host reduction (shared `cell_score` -> exact CPU==GPU). The traceback recovers the planted
  allele path. The conceptual leap from string DP to graph DP.
- **3.14 Metagenomic Classification** — alignment-free **k-mer hash-table** classification
  (one read per thread, grid-stride, shared `kmer_core.h` -> integer-exact CPU==GPU). 36/36
  classified reads recovered, 4 contaminants left unclassified. The Kraken idea, in miniature.
- **3.15 Hi-C / 3D Genome** — **ICE matrix balancing** (deterministic fixed-point atomic
  row-sum reduction over sparse COO nonzeros) + insulation-score **TAD** boundary calling.
  GPU bias matches CPU exactly; planted TAD borders recovered at bins 4 and 8.
- **3.16 Sequence Error Correction** — a **k-mer-spectrum / trusted-k-mer** corrector: an
  integer-atomic histogram for the spectrum + a one-thread-per-read greedy correction (shared
  core -> exact CPU==GPU). Synthetic errors 132 -> 39 (~70% removed). The BFC/Lighter approach.
- **3.17 CRISPR Guide Design** — a genome-wide per-window scan (one thread per genome
  position, 20-nt guide in constant memory, grid-stride) that PAM-gates (NGG), counts
  mismatches, and computes a **CFD off-target** score (shared `cfd_score.h` -> bit-exact).
  The CFD weights are a clearly-labeled synthetic position-only teaching model (not Doench).

All six are clearly-labeled **reduced-scope teaching versions** (synthetic data, labeled
synthetic), with production tools (Scanpy/Seurat, vg/GraphAligner, Kraken2/Centrifuge,
Cooler/cooltools, BFC/Lighter, CRISPOR/Cas-OFFinder) named in each `THEORY.md`.

## 4. How to build & run

```powershell
cd projects/03-genomics/3.13-pangenome-graph-alignment   # (or any of the six)
msbuild build/*.sln /p:Configuration=Release /p:Platform=x64
./demo/run_demo.ps1      # -> RESULT: PASS (GPU matches CPU)
```

No new CUDA libraries this batch — all custom kernels (graph/anti-diagonal DP, atomic hash
tables, COO reductions, constant-memory scans).

## 5. What to study here

Reading path: **3.14** (alignment-free hashing) -> **3.16** (k-mer spectrum + greedy fix) ->
**3.12** (normalize + KNN) -> **3.15** (ICE balancing + insulation TADs) -> **3.17** (CRISPR
PAM scan) -> **3.13** (the headline: SW over a DAG). Exercise: in **3.13**, add a branch to
the pangenome graph and confirm the alignment still finds the best path; in **3.15**, change
the insulation window and watch the called TAD boundaries shift.

## 6. Verification (lead-independent, not self-reports)

- ✅ **Boundaries:** only the 6 project folders + `docs/PATTERNS.md` changed; no artifacts.
- ✅ **Clean rebuild** (`/t:Rebuild`, fat arch list) of all 6 in both `Release|x64` and
  `Debug|x64`: **EXIT=0, 0 warnings, 0 errors** (12/12 builds).
- ✅ All 6 **demos PASS**: GPU==CPU (scRNA/pangenome/metagenome/error-correction/CRISPR exact;
  Hi-C bias 1e-9; CFD 1e-12).
- ✅ `verify_project.py` -> **DONE** for all 6 (comment ratios **0.68–1.09**).
- **Workflow:** 6 agents, ~1.05M agent tokens, 415 tool uses.
- **Environment:** RTX 2080 (SUPER), `sm_75`, CUDA 13.3, VS 2026 (`v145`).

## 7. Milestone note — 100 / 301 (one-third)

Three of 14 domains are now essentially built out (1 and 2 complete; 3 at 19/30) plus the 14
flagships. Patterns exercised so far span integer/DP alignment (string and **graph**), cuFFT
and cuBLAS/cuSOLVER library pipelines, ensemble-per-thread Monte Carlo / dynamics, stencils,
self-attention, and atomic hash tables. The ~6-project batch cadence with lead verification +
one push-note each has held steady through several account session-window resets.

## 8. Known limitations / TODOs

- All six are **reduced-scope teaching versions** (tiny cell matrices, small pangenome graphs,
  toy metagenomes, small Hi-C maps, short reads, a synthetic CFD model). Labeled synthetic.

## 9. Next push preview

Finish domain-3 Intermediates (`3.18` protein language models, `3.19` variant effect, `3.20`
HiFi overlap/polishing, `3.21` SV calling, `3.23` splice-aware alignment, `3.24` methylation
calling, `3.26` BAM sort/dedup, `3.27` suffix array/BWT/FM-index, `3.28` profile HMM, `3.29`
motif finding, `3.30` pangenome construction), completing **domain 3 (30/30)**. Then domain 4
(medical imaging). Same workflow, lead-verified.
