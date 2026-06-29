# 1.34 — Amyloid / Aggregation Propensity Prediction

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Drug%20Discovery%20%26%20Molecular%20Design-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 1: Drug Discovery & Molecular Design · Catalog ID `1.34`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

Many proteins clump into insoluble **amyloid fibrils** — the molecular event
behind Alzheimer's, Parkinson's, and ALS, and a leading failure mode of
antibody and peptide **biologic drugs**. Whether a protein aggregates is driven
largely by short, contiguous stretches of **β-aggregation-prone** residues
("aggregation-prone regions", APRs / "hot spots"). This project implements the
transparent, didactic core of the classic sequence predictors (TANGO,
AGGRESCAN, Zyggregator): map every residue to an **intrinsic aggregation
propensity**, **smooth** that signal with a sliding-window mean, **threshold** it
to find APRs, and **rank** a batch of proteins by their worst hot spot. The
smoothing is a 1-D convolution, so the GPU does it with the canonical
**shared-memory tiling** pattern, **one block per protein**, batched across the
whole set.

## What this computes & why the GPU helps

Production aggregation prediction spans two regimes: (1) **physics** —
coarse-grained / atomistic MD (MARTINI, GROMACS+PLUMED) that watches fibrils
nucleate, which needs µs–ms enhanced sampling only feasible on GPUs; and (2)
**sequence/structure scoring** — fast per-residue predictors (AGGRESCAN3D,
CamSol) and GNNs trained on experimental aggregation rates. This teaching project
implements regime (2)'s tractable core: a **per-residue propensity profile**.

For each protein we compute, at every residue `i`, the **centered windowed mean**
of intrinsic propensities over a window of `W` residues, then mark `i` as
aggregation-prone if that smoothed score crosses a threshold. Contiguous prone
residues form an APR.

**The parallel bottleneck:** the windowed mean is a **1-D sliding-window
convolution** evaluated at every residue of every protein. Two independent
parallelisms stack:

- **across proteins** — each sequence is scored independently (like 1.12's
  one-query-vs-many), so each protein gets its own **GPU block**;
- **across residues** — each smoothed value is an independent window average, so
  within a block each **thread** owns one residue.

A proteome-wide developability/liability screen runs this over **tens of
thousands** of sequences — that batch is what makes the problem GPU-bound. The
naive kernel re-reads each residue `W` times from global memory; the optimized
kernel stages a protein into **shared memory** once (tiling + halo) and reads the
window on-chip — the same lesson as flagship 7.10.

## The algorithm in brief

- **Encode** each one-letter residue to an index `0..20` on load.
- **Look up** each residue's intrinsic β-aggregation propensity from a fixed
  amino-acid scale (`src/propensity.h`).
- **Smooth**: `s[i] = mean of propensities over [i−h, i+h]` (clamped at the
  termini), `h = (W−1)/2`. This is the FIR/sliding-window convolution.
- **Threshold + segment**: residue `i` is prone if `s[i] ≥ THRESH`; contiguous
  prone residues form an APR. Report per protein the peak score & position, the
  prone-residue count, and the **longest contiguous APR**.
- **Rank** proteins by peak smoothed score (most aggregation-prone first).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/amyloid-aggregation-propensity-prediction.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/amyloid-aggregation-propensity-prediction.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\amyloid-aggregation-propensity-prediction.sln /p:Configuration=Release /p:Platform=x64
```

This project links only the CUDA runtime (`cudart_static.lib`) — no extra GPU
libraries — so the scaffolded `.vcxproj` is used unchanged.

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (CMake build)
```

The demo builds if needed, runs on `data/sample/amyloid_sample.fasta`, prints the
ranking and the top hit's smoothed profile, shows the GPU-vs-CPU agreement check,
and prints a timing line. You can also point the binary at any FASTA file:

```powershell
build\x64\Release\amyloid-aggregation-propensity-prediction.exe path\to\proteins.fasta
```

## Data

- **Sample (committed):** `data/sample/amyloid_sample.fasta` — four tiny
  **synthetic** proteins, designed so the correct ranking is known by
  construction (a strong β-core, a broad aliphatic stretch, a near-threshold
  alternating sequence, and a soluble charged control).
- **Full / real datasets:** `scripts/download_data.ps1` / `.sh` print pointers to
  AmyPro, WALTZ-DB 2.0, and EMDB (the binary accepts any FASTA).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: AmyPro — curated amyloidogenic sequence database
(https://amypro.net); WALTZ-DB 2.0 aggregation labels (https://waltzdb.switchlab.org);
ThT fluorescence assay aggregation kinetics datasets; EMDB fibril EM maps
(https://www.ebi.ac.uk/emdb/).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt): a
ranked table (the two designed aggregation-prone constructs on top, the soluble
control with **0** prone residues at the bottom), the top hit's smoothed profile,
and `RESULT: PASS`. The program computes the result on both the **GPU**
(`src/kernels.cu`) and a **CPU reference** (`src/reference_cpu.cpp`) and asserts:
the smoothed profiles & peak scores agree to ≤ `1e-5`, **and** every integer
field (peak position, prone count, longest APR) agrees **exactly**. That
agreement is the correctness guarantee — and because both sides call the same
`__host__ __device__` math (`src/propensity.h`), the smoothed values match to the
last bit (the demo's observed `max_abs_err` is `0`).

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads the FASTA batch, runs CPU + GPU, verifies, ranks, reports.
2. [`src/propensity.h`](src/propensity.h) — **the shared `__host__ __device__` physics**: the amino-acid scale, the propensity lookup, and the windowed mean (used identically by CPU and GPU — start here to understand *why* they agree exactly).
3. [`src/reference_cpu.h`](src/reference_cpu.h) / [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the data model, FASTA loader, ragged→padded batching, and the trusted serial scan.
4. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the one-block-per-protein, tiled-window idea.
5. [`src/kernels.cu`](src/kernels.cu) — the kernel (constant-memory scale, shared-memory tile, deterministic reduction) and host wrapper.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, the CUDA-event timer, and pure-C++ I/O helpers.

## Prior art & further reading

- **TANGO** (Fernandez-Escamilla et al., 2004) — statistical-mechanics β-aggregation predictor; the conceptual ancestor of the window+threshold approach used here.
- **AGGRESCAN** / **AGGRESCAN3D** (https://biocomp.chem.uw.edu.pl/A3D2/) — the per-residue *a3v* aggregation scale + a structure-aware version; study how a real scale is calibrated against experiment.
- **CamSol** (https://www-cohsoftware.ch.cam.ac.uk/index.php/camsolmethod) — solubility prediction; the flip side of aggregation, useful for antibody developability.
- **Zyggregator** / Pawar et al. (2005) — intrinsic aggregation-propensity scales; the spirit of `propensity.h`'s table.
- **WALTZ-DB 2.0** (https://waltzdb.switchlab.org) — experimentally labeled hexapeptides; the dataset to calibrate/validate a predictor against.
- **GROMACS + PLUMED** (https://github.com/gromacs/gromacs) — the GPU MD/metadynamics stack for the *physics* regime (fibril nucleation), the natural "next level" beyond sequence scoring.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Shared-memory tiled 1-D convolution, batched over sequences** (PATTERNS.md §1,
row "sliding-window / FIR / conv"; exemplar flagship 7.10). One **block per
protein**; one **thread per residue**; the per-residue propensities are staged
into **shared memory** once and the window mean reads on-chip; the tiny
amino-acid scale lives in **constant memory** (broadcast to every thread); the
per-protein reduction is done by a single thread to keep it **deterministic** (no
float atomics — PATTERNS.md §3). The catalog also lists the heavier physics
patterns (GPU MARTINI CG-MD, PLUMED metadynamics, GNN inference); those are
described in THEORY.md "Where this sits in the real world" as the path beyond
this teaching version.

## Exercises

1. **Calibrate against experiment.** Download WALTZ-DB 2.0, run the scanner on its
   hexapeptides, and sweep `THRESHOLD` to maximize agreement with the amyloid /
   non-amyloid labels (a tiny ROC curve). Which threshold is best?
2. **Swap the scale.** Replace `AA_PROPENSITY` with a published scale (e.g. the
   AGGRESCAN *a3v* values) and see how the ranking and APRs change. The pipeline
   is unchanged — only the model.
3. **Add a gatekeeper / charge correction.** Real predictors down-weight APRs
   flanked by charged residues (charges suppress aggregation). Add a per-residue
   charge term and combine it with the propensity before smoothing.
4. **Tile very long chains.** `AGG_MAX_LEN` caps the per-block tile at 1024
   residues. Extend the kernel to split a long sequence across multiple blocks
   (with a halo overlap) so titin-length proteins work — the production fix.
5. **Profile the tiling win.** Write a naive kernel that re-reads global memory
   `W` times per residue, then compare it to the tiled kernel on a large batch
   with Nsight Compute. Where does shared memory start to pay off?

## Limitations & honesty

- **Educational and synthetic.** The committed sequences are *invented*
  constructs (labeled synthetic everywhere), and the amino-acid propensity scale
  is an *illustrative* didactic table, not a calibrated, published scale. The
  output is **not** a validated aggregation prediction and **must not** inform any
  diagnostic, therapeutic, or formulation decision.
- **Sequence-only.** Real aggregation depends on 3-D structure, solvent exposure,
  concentration, pH, and dynamics. This model sees only the primary sequence —
  exactly the simplification that makes it fast and teachable, and exactly why it
  misses structure-driven APRs that AGGRESCAN3D or MD would catch.
- **A scan, not a simulation.** We do not simulate fibril nucleation; we score
  propensity. The physics regime (MARTINI/PLUMED on the GPU) is described in
  THEORY.md as the rigorous-but-expensive alternative.
- **Length cap.** Sequences longer than `AGG_MAX_LEN` (1024) are rejected with a
  clear message; multi-block tiling is left as Exercise 4.
