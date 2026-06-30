# 3.14 — Metagenomic Taxonomic Classification

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Genomics%2C%20Sequencing%20%26%20Bioinformatics-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 3: Genomics, Sequencing & Bioinformatics · Catalog ID `3.14`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

Given a pile of unlabeled DNA **reads** from a mixed microbial sample, this project
answers *"who is in here?"* — it assigns each read to a **taxon** (species) and
reports the community's **abundance profile**. It does this the modern,
alignment-free way: every reference genome is chopped into short **k-mers** stored
in a hash map `k-mer → taxon`, and each read is classified by looking up its k-mers
and letting them **vote**. The look-up is embarrassingly parallel — one read per GPU
thread — which is exactly the bottleneck that GPU classifiers (MetaCache-GPU)
accelerate for real-time diagnostics. Everything runs on a tiny **synthetic**
community so you can watch it recover the right answer offline.

## What this computes & why the GPU helps

Metagenomic classification assigns every sequencing read to a taxon by matching
k-mers against a database of reference genomes (Kraken2 uses an exact k-mer LCA hash
map; Centrifuge uses an FM-index). At clinical sequencing throughput (millions of
reads/minute), the **hash look-up** is the bottleneck. It is also perfectly
parallel: classifying one read is independent of every other read.

**The parallel bottleneck:** for each read we slide a k-mer window and **probe the
reference hash table once per k-mer**. With millions of reads × dozens of k-mers
each, that is hundreds of millions of independent table probes — so we give **each
read its own GPU thread** (a grid-stride loop covers any number of reads). This is
the *"score one query vs N items, each independent"* GPU pattern
(`docs/PATTERNS.md` §1), the same family as flagship `1.12`.

## The algorithm in brief

- **Encode** each base in 2 bits and roll a k-mer window across each sequence in O(1).
- **Canonicalize** each k-mer to `min(forward, reverse-complement)` so the table is
  strand-agnostic.
- **Build** an open-addressing (linear-probing) hash table `canonical k-mer → taxon`
  from the reference genomes (host, once).
- **Classify** each read on the GPU: probe the table for every k-mer, tally per-taxon
  votes, assign the **argmax** taxon (lowest-id tie-break), or *unclassified* if no
  k-mer matched.
- **Verify** the GPU's integer taxon ids against the CPU reference **exactly**, and
  report accuracy vs the synthetic ground truth.

See [`THEORY.md`](THEORY.md) for the math, complexity, and GPU-mapping depth.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3** (the repo's
ratified standard — see [`docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md)).

1. Open `build/metagenomic-taxonomic-classification.sln` in Visual Studio 2026.
2. Select **`Release | x64`**.
3. **Build** (Ctrl+Shift+B). The `.exe` lands in `build/x64/Release/`.

Command line (Developer PowerShell or via the demo script):

```powershell
& "C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\MSBuild.exe" `
  build\metagenomic-taxonomic-classification.sln /p:Configuration=Release /p:Platform=x64 /m
```

Linux/macOS learners can use the optional `CMakeLists.txt` (`cmake -S . -B build/cmake
-DCMAKE_BUILD_TYPE=Release && cmake --build build/cmake`); the VS solution is the
required deliverable.

## Run the demo

One command builds (if needed), runs on the committed sample, and checks the output:

```powershell
powershell -ExecutionPolicy Bypass -File demo\run_demo.ps1     # Windows
```
```bash
./demo/run_demo.sh                                             # Linux/macOS (CMake)
```

It prints the abundance profile, the GPU-vs-CPU agreement, and `PASS`/`FAIL`.

## Data

The committed sample `data/sample/metagenome_sample.txt` is **synthetic** (random
DNA; the species names are illustrative labels, not real genomes). It contains 5
reference "genomes" and 40 reads, generated deterministically by
`scripts/make_synthetic.py` (seed 7). The reads are an uneven mix across taxa plus a
few random "contaminant" reads, so the recovered profile is interesting and 4 reads
correctly come back *unclassified*.

- Regenerate / scale: `python scripts/make_synthetic.py [--reads N]`.
- Real data: `scripts/download_data.ps1` / `.sh` print how to build a real reference
  database from **NCBI RefSeq** genomes and classify **CAMI / HMP / SRA** reads — no
  downloads are performed automatically and no credentialed access is bypassed.
- Provenance, format, license, and checksum: [`data/README.md`](data/README.md).

## Expected output

```
3.14 -- Metagenomic Taxonomic Classification
k-mer classification: 40 reads vs 2930 reference 15-mers (5 taxa)
taxonomic abundance profile (reads assigned per taxon):
  taxon 1  Escherichia_coli          12 reads
  taxon 2  Staphylococcus_aureus     10 reads
  taxon 3  Pseudomonas_aeruginosa    7 reads
  taxon 4  Bacteroides_fragilis      3 reads
  taxon 5  Lactobacillus_casei       4 reads
  (none)   unclassified              4 reads
accuracy on classified reads: 36/36 correct
RESULT: PASS (GPU taxon ids match CPU exactly; 40/40 reads agree)
```

**How success is checked.** The taxon ids are integers, and the CPU and GPU both run
the identical `__host__ __device__` `classify_read` core, so they must agree
**exactly** — the verification tolerance is `0` (no floating point anywhere). The
demo diffs the deterministic **stdout** against `demo/expected_output.txt`; timings
go to **stderr** (shown, not diffed). The `36/36 correct` line is a second,
science-level check against the synthetic ground truth.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — the 5-step shape: load → CPU → GPU → verify → report.
2. [`src/kmer_core.h`](src/kmer_core.h) — **the heart**: the shared
   `__host__ __device__` k-mer encoding, canonicalization, hash, table probe, and
   the per-read vote. CPU and GPU both call this, which is why they agree exactly.
3. [`src/kernels.cuh`](src/kernels.cuh) → [`src/kernels.cu`](src/kernels.cu) — the
   GPU harness: one thread per read (grid-stride), calling `classify_read`.
4. [`src/reference_cpu.h`](src/reference_cpu.h) →
   [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the data model, the dataset
   loader, the hash-table builder, and the serial baseline.
5. `src/util/` — shared `CUDA_CHECK` error macros and CUDA-event timing.

## Prior art & further reading

From the catalog's *Starter Repos / Tools* (study these; do not copy — we
reimplement didactically):

- **Kraken2** (https://github.com/DerrickWood/kraken2) — the canonical CPU
  classifier; minimizer + **LCA** hash map. Our table is the same idea, minus
  minimizers and the taxonomy tree.
- **MetaCache-GPU** (arXiv:2106.08150) — GPU k-mer classification with cuckoo/robin-
  hood tables and a persistent kernel; the production form of what we prototype.
- **Centrifuge** (https://github.com/DaehwanKimLab/centrifuge) — FM-index backward
  search instead of a hash; a different memory/speed trade-off.
- **Bracken** (https://github.com/jenniferlu717/Bracken) — Bayesian abundance
  re-estimation downstream of Kraken2; refines the raw read counts we produce.

## Exercises

1. **Minimizers.** Index only the minimizer of each window (the smallest k-mer in a
   small neighborhood) instead of every k-mer. How much does the table shrink, and
   what happens to accuracy on the sample?
2. **Confidence threshold.** Kraken2 calls a read only if its top taxon wins by a
   margin. Add a `min_hit_fraction` and route low-confidence reads to *unclassified*.
3. **LCA.** Add a tiny 2-level taxonomy (species → genus) and, when a k-mer is shared
   by two species, map it to their genus. Re-run and watch shared-k-mer reads move up.
4. **Scale it.** `python scripts/make_synthetic.py --reads 2000000`, rebuild, and
   compare CPU vs GPU time — find the read count where the GPU starts winning.
5. **GPU build.** Move `build_database` onto the device using `atomicCAS` insertions
   (the MetaCache-GPU approach); verify the table matches the host-built one.

## Limitations & honesty

- **Synthetic data.** Random DNA with illustrative species labels — not real
  genomes, no clinical validity. Labeled synthetic everywhere.
- **Reduced scope (teaching version).** We use **exact k-mers** (no minimizers), a
  **flat** taxon set with *first-writer-wins* on shared k-mers (no LCA tree), and a
  **host-built** table (we parallelize only the look-up, which is the bottleneck).
  `THEORY.md` describes each production difference.
- **Tiny demo, honest timing.** On 40 reads the GPU kernel is *slower* than the CPU —
  launch/copy overhead dominates. The GPU's edge appears at millions of reads; the
  timing line is a teaching artifact, never a benchmark claim.
- **Fixed `k = 15` and `MAX_TAXA`.** Compile-time constants in `kmer_core.h`; k-mers
  mapping to taxon ids ≥ `MAX_TAXA` would be ignored by the vote.
