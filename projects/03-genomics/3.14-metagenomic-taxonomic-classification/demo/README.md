# Demo — 3.14 Metagenomic Taxonomic Classification

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed synthetic sample (`data/sample/metagenome_sample.txt`),
   which holds 5 reference "genomes" and 40 reads.
3. **Classify** every read on both the **CPU** (`reference_cpu.cpp`) and the **GPU**
   (`kernels.cu`) by matching its 15-mers against the reference hash table, and
   **verify** the two assign byte-identical taxon ids (integer ids → tolerance 0).
4. **Report** a taxonomic abundance profile and accuracy vs the synthetic ground
   truth on stdout; **time** the kernel (CUDA events) on stderr — a *teaching
   artifact*, not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries timing (which varies run to run), so it is shown but never diffed.

## Expected result

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

## How to read it

- The **abundance profile** is the headline scientific output — the count of reads
  assigned to each taxon. It recovers the simulated community (taxon 1 dominant
  down to taxon 5 rare), which is exactly what a metagenomic profiler is for.
- **4 reads are unclassified**: those are the pure-random "contaminant" reads in the
  sample; none of their 15-mers are in the reference table, so they correctly fall
  into the unclassified bin.
- **36/36 correct** means every read that *was* classified got the taxon it was
  simulated from — a clean recovery because the random reference genomes share
  essentially no 15-mers by chance.
- **RESULT: PASS** confirms the GPU and CPU produced identical taxon ids for all 40
  reads. Because the two share the `__host__ __device__` `classify_read()` core,
  the agreement is exact (not "within a tolerance").

> The `species` names are illustrative labels for **synthetic random DNA**, not real
> genomes. Educational only — not for any clinical use.
