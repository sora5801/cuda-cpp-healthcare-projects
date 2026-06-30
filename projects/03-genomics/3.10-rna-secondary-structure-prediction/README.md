# 3.10 — RNA Secondary-Structure Prediction

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Genomics%2C%20Sequencing%20%26%20Bioinformatics-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 3: Genomics, Sequencing & Bioinformatics · Catalog ID `3.10`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

An RNA molecule is a single strand that folds back on itself: complementary bases
pair up (A–U, G–C, and the wobble G–U) to form the stems and hairpin loops that
give the molecule its shape and function. This project predicts that **secondary
structure** with the **Nussinov algorithm** — the classic teaching model that
finds the folding with the **maximum number of (non-crossing) base pairs** via a
cubic-time dynamic program. The interesting part for us is the *parallelism*: the
DP table has the same diagonal-dependency structure as sequence alignment, so we
fill it on the GPU as an **anti-diagonal wavefront**, then verify the GPU's table
cell-for-cell against a plain serial CPU version.

## What this computes & why the GPU helps

We fill a matrix `M`, where `M[i][j]` is the most base pairs achievable in the
sub-sequence `s[i..j]`. The full sequence's answer is `M[0][n-1]`, and a
**traceback** turns the optimal table into a dot-bracket structure like
`((((((....))))))..`. Real RNA folding (the **Zuker** algorithm) minimises *free
energy* with the same `O(n³)` time / `O(n²)` space DP; for long RNAs (rRNA,
lncRNA > 10 kb) the cubic cost is punishing on a CPU.

**The parallel bottleneck:** filling the `O(n²)` DP cells, each of which does an
`O(n)` "bifurcation" scan. A cell `M[i][j]` depends only on cells of **smaller
span** `L = j − i`. So every cell on one span diagonal is independent and can be
computed **simultaneously**. We sweep spans `L = 1 … n−1`; each span is one
parallel kernel launch. GPU RNAfold implementations report ~14× speedups on long
sequences exploiting exactly this wavefront.

## The algorithm in brief

- **Nussinov recurrence** (max base pairs): for each `M[i][j]`, take the best of
  *i unpaired*, *j unpaired*, *i pairs with j* (`+1` if allowed), and the
  **bifurcation** `max over k of M[i][k] + M[k+1][j]`.
- **Pairing rule:** A–U, G–C, G–U (wobble), with a minimum hairpin loop of 3
  unpaired bases.
- **Anti-diagonal wavefront parallelism:** fill the upper triangle span by span.
- **Traceback:** recover one optimal dot-bracket structure (serial, on the host).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation, including how Nussinov relates to the energy-minimising Zuker /
McCaskill models and the `O(n)` LinearFold approximation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/rna-secondary-structure-prediction.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/rna-secondary-structure-prediction.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\rna-secondary-structure-prediction.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/rna_sample.fasta`, prints the
predicted structure, shows the GPU-vs-CPU agreement check, and prints a timing
line. You can fold any sequence by passing a FASTA/text file as the first
argument: `... .exe path\to\my_rna.fasta`.

## Data

- **Sample (committed):** `data/sample/rna_sample.fasta` — a tiny, **synthetic**
  18-nt hairpin so the demo runs with zero downloads. Its optimal fold (6 base
  pairs) is a known answer, making the demo a real correctness check.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` point at real public RNA
  databases (Rfam, RNAcentral, PDB, ArchiveII) and can fetch one Rfam family on
  demand; they never bypass any site's terms.
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: Rfam — RNA family alignments and secondary structures
(<https://rfam.org/>); RNAcentral — comprehensive RNA sequence database
(<https://rnacentral.org/>); PDB RNA structures (<https://www.rcsb.org/>);
ArchiveII benchmark — curated RNA secondary structures.

## Expected output

Success looks like `demo/expected_output.txt`:

```
3.10 -- RNA Secondary-Structure Prediction (Nussinov)
RNA length n = 18  (alphabet ACGU, min hairpin loop = 3)
sequence : GGGCGCAAAAGCGCCCAU
structure: ((((((....))))))..
max base pairs = 6
RESULT: PASS (GPU matrix matches CPU exactly)
```

The program fills the DP table on the **GPU** (`src/kernels.cu`) and on a **CPU
reference** (`src/reference_cpu.cpp`) and asserts the two integer matrices are
**identical, cell for cell** — that exact agreement is the correctness guarantee.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads the RNA, runs CPU + GPU, verifies, reports.
2. [`src/reference_cpu.h`](src/reference_cpu.h) — the data model, the pairing rule,
   and the **shared `__host__ __device__` recurrence** (`nussinov_cell`) used by
   *both* backends so their results match exactly.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the wavefront idea
   (with an ASCII picture of the span diagonals).
4. [`src/kernels.cu`](src/kernels.cu) — the per-span kernel and the host sweep.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline +
   loader + traceback.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, the CUDA-event timer, host I/O.

## Prior art & further reading

- **ViennaRNA / RNAfold** (`https://www.tbi.univie.ac.at/RNA/`) — the standard
  energy-minimising folder (Zuker) and the McCaskill partition function; study its
  Turner energy model to see what Nussinov's "+1 per pair" abstracts away.
- **CUDA RNAfold** (`https://www.biorxiv.org/content/10.1101/298885v1.full`) —
  GPU-parallelised Vienna RNAfold (~14× on long RNAs) using exactly this
  anti-diagonal wavefront with shared-memory tiling.
- **LinearFold** (`https://github.com/LinearFold/LinearFold`) — an `O(n)`
  beam-search approximation; lends itself to GPU batching of many short RNAs.
- **LinearAlifold** (`https://github.com/LinearFold/LinearAlifold`) — consensus
  structure across an alignment.
- **EternaFold** (`https://github.com/eternagame/EternaFold`) — an ML-trained
  folding model.

Study these to learn the production approach; **do not copy code wholesale** —
this project reimplements the simplest correct teaching version (Nussinov) and
credits the sources (CLAUDE.md §2).

## CUDA pattern used here

**Anti-diagonal wavefront** over a dynamic-programming table (docs/PATTERNS.md
§1; exemplified by flagship `3.01` Smith-Waterman). A custom kernel fills one
span diagonal `L = j − i` per launch — all cells on a span are independent — and
the per-cell math is a shared `__host__ __device__` function (PATTERNS §2) so the
CPU and GPU produce bit-identical integers (exact verification, PATTERNS §4).

## Exercises

1. **Energy-aware scoring.** Give G–C pairs a weight of 3, A–U a weight of 2, and
   G–U a weight of 1 (toward a Turner-style model). The recurrence and the
   wavefront are unchanged — only `pair_score` changes. How does the optimal
   structure shift?
2. **Stress the wavefront.** Generate a length-200 RNA
   (`python scripts/make_synthetic.py --random 200`) and watch the CPU/GPU timing
   gap change as the early spans grow wide. Where does the GPU start to win?
3. **Shared-memory tiling.** The bifurcation scan re-reads a whole row/column of
   `M` from global memory per cell. Stage that row/column into shared memory for
   the block and measure the bandwidth saving (this is what CUDA RNAfold does).
4. **Base-pair probabilities.** Replace the `max` with a `sum` and floating-point
   Boltzmann weights to compute the McCaskill **partition function** — then
   discuss why that breaks the exact-integer verification (PATTERNS §4).
5. **Pseudoknots.** Nussinov forbids crossing pairs. Add one specific crossing
   pattern and explain why the general pseudoknot problem becomes NP-hard.

## Limitations & honesty

- **This is the Nussinov teaching model, not production folding.** It maximises a
  *count of base pairs*; real folders (ViennaRNA, RNAstructure) minimise
  *thermodynamic free energy* with the Turner nearest-neighbour parameters, which
  is far more accurate. Nussinov structures are illustrative, not authoritative.
- **No pseudoknots, no multi-loop penalties, no coaxial stacking** — all of which
  real models handle.
- **The committed sample is synthetic** (a designed hairpin), labeled synthetic
  everywhere. It is engineered to have a clear, checkable optimal fold.
- **Timing is a teaching artifact, not a benchmark.** On a short RNA the per-span
  launches dominate and the GPU can be *slower* than the CPU; the wavefront pays
  off on long sequences and batched workloads (see THEORY §7).
- **Not for clinical use.** Nothing here informs diagnosis or treatment.
