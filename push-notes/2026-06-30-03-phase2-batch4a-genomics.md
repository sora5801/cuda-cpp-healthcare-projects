# Push 2026-06-30 #03 -- phase2 batch4a genomics

> Push-note (CLAUDE.md section 7.1). First Phase-2 batch in **Domain 3 (Genomics, Sequencing
> & Bioinformatics)**: 6 Beginner projects, each worker-built and lead-verified.

## 1. Summary

The build-out crosses into its **third domain**. Six **domain-3 (genomics)** Beginner
projects are complete, taking the collection to **82 -> 88 / 301 (29.2%)** and domain 3 to
**7/30** (the flagship `3.01` Smith-Waterman was already done). This batch covers the core
of a sequencing pipeline: short-read mapping, variant calling, nanopore basecalling, k-mer
sketching, BLAST homology search, and RNA-seq quantification. Because genomics is almost
entirely **integer / bit / DP** work, **five of the six verify CPU==GPU bit-exactly** — a
nice contrast to the floating-point physics of domains 1-2. Each was built in its own folder
by one worker and re-verified by the lead.

## 2. What changed

Six new fully-implemented projects under `projects/03-genomics/`:

- [`3.02` Short-Read Mapping / Alignment](../projects/03-genomics/3.02-short-read-mapping-alignment)
- [`3.03` Variant Calling Acceleration](../projects/03-genomics/3.03-variant-calling-acceleration)
- [`3.04` Nanopore Basecalling](../projects/03-genomics/3.04-nanopore-basecalling)
- [`3.06` k-mer Counting & Minimiser Sketching](../projects/03-genomics/3.06-k-mer-counting-minimiser-sketching)
- [`3.07` BLAST-Style Homology Search](../projects/03-genomics/3.07-blast-style-homology-search)
- [`3.22` RNA-seq Quantification / Pseudo-alignment](../projects/03-genomics/3.22-rna-seq-quantification-pseudo-alignment)

`docs/STATUS.md` -> these 6 marked **done** (88/301). `CHANGELOG.md` indexed.

## 3. New projects (didactic blurb each)

- **3.02 Short-Read Mapping** — a **seed-and-extend** mapper: a sorted 2-bit k-mer index +
  binary-search seeding + ungapped extension, one thread per read, sharing an integer scoring
  core with the CPU (exact, tol=0). The GPU shape behind minimap2/BWA, in miniature.
- **3.03 Variant Calling** — the **PairHMM forward algorithm** (the compute core of
  GATK HaplotypeCaller): one thread per (read, haplotype) pair fills its own DP table, shared
  double-precision recurrence -> CPU==GPU ~1.8e-15, all 8 synthetic reads assigned to the
  truth haplotype. (nvcc gotcha fixed: literal `1.0/0.0` infinity -> `-INFINITY`/numeric_limits.)
- **3.04 Nanopore Basecalling** — reduced to the **CTC greedy decode** (the NN is out of
  scope): turn a posterior matrix into DNA bases, one read per thread, shared `ctc_core.h` ->
  exact CPU==GPU. Planted sequences (including homopolymers) are recovered exactly.
- **3.06 k-mer Counting & Sketching** — a **device open-addressing hash table** (atomicCAS
  claim + integer atomicAdd tally, deterministic) for canonical k-mer counting, a
  sliding-window-minimum **minimiser** kernel, and **bottom-s MinHash** Jaccard estimation.
  All exact CPU==GPU. The atomic hash table is the thing to read.
- **3.07 BLAST Homology Search** — **seed-filter-extend**: a k-mer prefilter + gapless
  **X-drop** extension (BLOSUM62 in constant memory), one thread per DB sequence, integer
  scoring -> exact CPU==GPU. The classic BLAST heuristic on the GPU.
- **3.22 RNA-seq Quantification** — the **pseudo-alignment EM** abundance estimator (kallisto-
  style): one thread per equivalence class (E-step split + fixed-point integer atomicAdd
  M-step), shared core -> exact CPU==GPU; EM recovers the planted ground truth to L1~0.

All six are clearly-labeled **reduced-scope teaching versions** (synthetic reads/refs, labeled
synthetic), with production tools (BWA-MEM/minimap2, GATK/DeepVariant, Dorado/Bonito, KMC/
Mash, BLAST+/DIAMOND, kallisto/salmon) named in each `THEORY.md`.

## 4. How to build & run

```powershell
cd projects/03-genomics/3.06-k-mer-counting-minimiser-sketching   # (or any of the six)
msbuild build/*.sln /p:Configuration=Release /p:Platform=x64
./demo/run_demo.ps1      # -> RESULT: PASS (GPU matches CPU)
```

No new CUDA libraries this batch — all custom kernels (integer DP, atomic hash table, constant
memory, grid-stride).

## 5. What to study here

Reading path: **3.04** (simplest: a per-read greedy decode) -> **3.02** / **3.07** (seed-and
-extend, the shared backbone of mapping and homology search) -> **3.06** (the atomic hash
table + sketching) -> **3.22** (EM with integer-atomic reductions) -> **3.03** (PairHMM DP, a
floating-point cousin of Smith-Waterman 3.01). Exercise: in **3.06**, change the minimiser
window w and watch the sketch size vs. sensitivity trade-off; in **3.03**, add a third
candidate haplotype and confirm the reads still map to the truth.

## 6. Verification (lead-independent, not self-reports)

- ✅ **Boundaries:** only the 6 project folders changed; no shared/root file; no artifacts.
- ✅ **Clean rebuild** (`/t:Rebuild`, fat arch list) of all 6 in both `Release|x64` and
  `Debug|x64`: **EXIT=0, 0 warnings, 0 errors** (12/12 builds).
- ✅ All 6 **demos PASS**: GPU==CPU (mapping/basecall/kmer/BLAST exact; variant 1.8e-15;
  RNA-seq abundances match).
- ✅ `verify_project.py` -> **DONE** for all 6 (comment ratios **0.81–1.00**).
- **Workflow:** 6 agents, ~1.03M agent tokens, 433 tool uses (relaunched cleanly after a
  session-window reset; the first attempt was killed mid-run by the usage limit).
- **Environment:** RTX 2080 (SUPER), `sm_75`, CUDA 13.3, VS 2026 (`v145`).

## 7. Known limitations / TODOs

- All six are **reduced-scope teaching versions**: tiny references/read sets, a CTC decode
  without the neural net, a small equivalence-class set for EM. Labeled synthetic; production
  scale is described in each THEORY.md.

## 8. Next push preview

Continue domain-3 Beginners (`3.25` BQSR), then the Intermediate tier (`3.5` assembly, `3.8`
MSA, `3.9` phylogenetics, `3.10` RNA structure, `3.11` GWAS, `3.12` scRNA-seq, …) in
~6-project batches. Same workflow, lead-verified, one push-note per batch.
