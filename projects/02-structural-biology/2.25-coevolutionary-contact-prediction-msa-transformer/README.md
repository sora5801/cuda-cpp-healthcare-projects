# 2.25 — Coevolutionary Contact Prediction & MSA Transformer

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Structural%20Biology%20%26%20Protein%20Science-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 2: Structural Biology & Protein Science · Catalog ID `2.25`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

When a protein folds, pairs of residues that touch in 3-D tend to **mutate
together** over evolution: a change at one position is compensated by a change at
its contacting partner. Line up many homologs of a protein in a **Multiple
Sequence Alignment (MSA)** — N sequences × L columns — and those contacting
**columns become statistically dependent**. This project measures that dependence
for **every pair of columns** using **Mutual Information (MI)**, cleans it up with
the standard **Average Product Correction (APC)**, and ranks the strongest pairs
as predicted **residue contacts** — the signal that drives modern protein
structure prediction. Each of the L(L−1)/2 column pairs is an independent
reduction over the N sequences, so we give **one GPU thread per pair**. The demo
runs on a synthetic MSA with four *planted* contacts and recovers all four at the
top of the ranking, with the GPU result verified bit-for-bit against a CPU
reference.

## What this computes & why the GPU helps

Coevolutionary analysis of MSAs (correlated mutations between residue positions) reveals protein contact maps that drive structure prediction. EVcouplings uses PLMC (pseudolikelihood-maximized direct coupling analysis) — an L×L matrix inversion and optimization problem where L is sequence length. GPU acceleration via direct CUDA implementation or PyTorch autograd parallelizes the DCA learning over position pairs. MSA Transformer (ESM-MSA-1b) processes MSA rows and columns via tied axial attention on GPU, producing contact predictions and rich evolutionary embeddings for downstream tasks.

**The parallel bottleneck:** the coevolution score is an **L × L matrix**, one
entry per column pair. For each pair (i, j) we scan all N sequences to build a
joint amino-acid count table, then reduce it to one MI number. That is
**L(L−1)/2 independent reductions over N rows** — the dominant cost, and
embarrassingly parallel. We map **one column pair → one GPU thread**: every thread
builds its own small joint-count table in local memory and computes its MI with
no communication or synchronization. (Production tools like CCMpred parallelize
the *same* per-pair structure for their fancier DCA gradients.) This is the
"score all pairs, each independent" pattern from `docs/PATTERNS.md` §1 — the same
shape as flagship `1.12` (Tanimoto) and `12.01` (spectral search).

## The algorithm in brief

- **Tokenize** the MSA: each residue → an integer in `[0, 21)` (20 amino acids + gap).
- **Marginals:** count each symbol's frequency in each column (once, reused).
- **Per pair (i, j):** build the Q×Q **joint count** table over the N sequences,
  then compute **Mutual Information** `MI(i,j) = Σ p_ab · ln(p_ab / (p_i(a)·p_j(b)))`.
- **APC correction:** `score(i,j) = MI(i,j) − MIcol(i)·MIcol(j)/MImean` removes the
  per-column entropic/phylogenetic background (Dunn et al. 2008).
- **Predict:** rank column pairs by corrected score; the top pairs are the
  predicted contacts.

Full pairwise Direct Coupling Analysis (PLMC/mpDCA) and the MSA Transformer go
further; see [THEORY.md](THEORY.md) for the science → math → algorithm →
GPU-mapping derivation and where this sits relative to those methods.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/coevolutionary-contact-prediction-msa-transformer.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/coevolutionary-contact-prediction-msa-transformer.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\coevolutionary-contact-prediction-msa-transformer.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/coevolution_msa.fasta`, prints the
top predicted contacts, shows the GPU-vs-CPU agreement check, and prints a timing
line. You can also point the program at your own aligned-FASTA MSA:

```powershell
build\x64\Release\coevolutionary-contact-prediction-msa-transformer.exe path\to\family.fasta
```

## Data

- **Sample (committed):** `data/sample/coevolution_msa.fasta` — a tiny **synthetic**
  MSA (400 sequences × 24 columns) with four planted coevolving column pairs, so
  the demo runs offline and the result is verifiable.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print pointers to real
  family MSAs (nothing is auto-downloaded); `scripts/make_synthetic.py` makes a
  larger synthetic MSA.
- **Provenance & license:** see [data/README.md](data/README.md). The sample is
  synthetic and carries no biological/clinical meaning.

Catalog dataset notes: UniRef50/UniRef90 for MSA construction (https://www.uniprot.org); Pfam MSA database (https://pfam.xfam.org); EVcouplings benchmark contact sets (https://github.com/debbiemarkslab/EVcouplings); CASP14 contact prediction benchmarks (https://predictioncenter.org).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt): the
four planted contacts `(4,5) (9,16) (3,22) (6,19)` appear as ranks **#1–#4** (APC
≈ 1.3–1.4), an order of magnitude above the best decoy (≈ 0.13), and the line
`RESULT: PASS`. The program computes the raw MI matrix on both the **GPU**
(`src/kernels.cu`) and a **CPU reference** (`src/reference_cpu.cpp`) and asserts
they agree to within `1e-9` nats — in practice they match to ~`4e-16` (machine
precision), because both derive MI from identical integer counts and evaluate the
same shared `cv_mi_from_counts`. That agreement is the correctness guarantee.

## Code tour

Read in this order:

1. [`src/coevolution.h`](src/coevolution.h) — the shared `__host__ __device__` core:
   the alphabet, the tokenizer, and `cv_mi_from_counts` (the one true MI formula).
2. [`src/main.cu`](src/main.cu) — loads the MSA, runs CPU + GPU, verifies, ranks
   and prints the predicted contacts.
3. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline:
   FASTA loader, the pair loop, and the APC correction.
4. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the one-thread-per-pair idea.
5. [`src/kernels.cu`](src/kernels.cu) — the `mi_pairs_kernel` and host wrapper.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, CUDA-event timer, host I/O helpers.

## Prior art & further reading

- **EVcouplings** (<https://github.com/debbiemarkslab/EVcouplings>) — DCA-based
  coevolution (PLMC). Study how raw MI is replaced by *direct* couplings that
  subtract indirect (transitive) correlations — the key step beyond this project.
- **CCMpred** (<https://github.com/soedinglab/CCMpred>) — GPU-accelerated DCA with
  custom CUDA kernels. Study how it parallelizes the per-pair pseudolikelihood
  gradient (the same independent-pairs structure we use here).
- **ESM-MSA-1b / MSA Transformer** (<https://github.com/facebookresearch/esm>) —
  the deep-learning route: tied axial (row/column) attention over the MSA. Study
  how attention generalizes "pairwise column statistics" into a learned model.
- **HHpred** (<https://toolkit.tuebingen.mpg.de/tools/hhpred>) — profile-profile
  alignment, useful upstream for building a good MSA.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

CCMpred custom CUDA kernels for DCA gradient computation; cuBLAS for L×L coupling matrix products; PyTorch CUDA axial attention for MSA Transformer; GPU-parallel MSA column featurization.

In this teaching version the pattern is **"many independent jobs"**: a 2-D grid of
threads where thread (i, j) computes one matrix entry MI(i, j) over the N
sequences. No shared memory, no atomics, no cuBLAS — every thread writes disjoint
output cells, so the result is deterministic and equals the CPU reference exactly
(see [THEORY.md](THEORY.md) §GPU-mapping and §Verification). The fancier
cuBLAS/PyTorch routes named above are described in THEORY's "real world" section.

## Exercises

1. **Direct vs. indirect coupling.** MI flags *all* correlated pairs, including
   indirect ones (i–j and j–k coupled ⇒ i–k looks coupled). Add a second metric
   that subtracts indirect effects — even a simple covariance-matrix inversion
   ("mean-field DCA") — and compare its top contacts to MI+APC.
2. **Gap handling.** We treat gap as just another symbol. Down-weight gappy
   columns, or skip sequences that are >50% gaps, and see how the ranking changes.
3. **Sequence reweighting.** Real MSAs are phylogenetically biased (many near-
   identical sequences). Weight each sequence by `1 / (#sequences within 80%
   identity)` and use the weights in the counts. This is the single biggest
   accuracy lever in real DCA.
4. **Triangular work packing.** Half our threads (i ≥ j) return immediately.
   Remap a 1-D thread index to the upper-triangular pair `(i, j)` so every thread
   does real work, and measure the occupancy/throughput change.
5. **Shared-memory marginals.** For large L, cache a block's column marginals in
   shared memory. Profile with Nsight Compute and report the change in global-load
   traffic.

## Limitations & honesty

- **Teaching scope.** This is the *foundational* coevolution estimator (pairwise
  MI + APC), not full DCA. It does **not** disentangle direct from indirect
  couplings, so on real data it under-performs PLMC/EVcouplings and far
  under-performs the MSA Transformer. THEORY.md §"Where this sits in the real
  world" explains the gap.
- **Synthetic data.** The committed sample is synthetic, with covariation planted
  by a fixed codebook. It is engineered so the answer is known and the demo is
  verifiable; it is **not** a real protein family and implies **nothing**
  biological or clinical.
- **No sequence reweighting / pseudocounts.** Real pipelines reweight redundant
  sequences and add pseudocounts to stabilize sparse counts; we omit both for
  clarity (they are exercises).
- **Timing is a teaching artifact.** With L=24 the CPU wins (276 pairs cannot
  amortize launch + copy overhead). The GPU's advantage appears only at realistic
  L (hundreds to thousands). We never present timing as a benchmark claim.
