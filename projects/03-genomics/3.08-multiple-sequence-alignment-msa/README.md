# 3.8 — Multiple Sequence Alignment (MSA)

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Genomics%2C%20Sequencing%20%26%20Bioinformatics-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 3: Genomics, Sequencing & Bioinformatics · Catalog ID `3.8`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

Align **N DNA sequences at once** with the classic **progressive** recipe used by
ClustalW / MAFFT: first score *every pair* of sequences, then use those scores to
pick a guide and fold the sequences together into one alignment. The expensive
first step — an **N×N pairwise alignment matrix** (`O(N²)` independent
Needleman-Wunsch alignments) — is the part the GPU accelerates, and it maps onto
the cleanest possible pattern: **one CUDA thread block per pair**. This is the
deliberate sequel to `3.01` (a *single* alignment, parallelised *within* by a
wavefront); here the parallelism is *across* the many independent alignments.

## What this computes & why the GPU helps

Progressive MSA has three stages: (1) build a pairwise **distance matrix** from
all-vs-all alignment scores; (2) choose a **guide** (here the *center-star*
sequence, the one closest to all others); (3) **progressively** align everything
to that center into one multi-row alignment. Stage 1 is `N(N-1)/2` Needleman-
Wunsch alignments, each an `O(L²)` dynamic-programming fill — for large N this is
the wall-time bottleneck (the catalog notes a reported **~6× speedup** for the
MAFFT-PartTree distance phase on GPU). Those alignments are **mutually
independent**, so the GPU runs them in parallel: **block `p` scores pair `p`**,
keeping that pair's two rolling DP rows in fast **shared memory**. Stages 2–3 are
cheap host bookkeeping and are *not* the GPU lesson.

**The parallel bottleneck** is the `O(N²)` distance-matrix phase; we parallelize
it across pairs (one thread block per pair).

## The algorithm in brief

- **Stage 1 — distance matrix:** for every pair `(a,b)`, the Needleman-Wunsch
  global-alignment score `H[la][lb] = max(diag+s, up+gap, left+gap)`; normalise to
  a distance `D = 1 − score/self`.
- **Stage 2 — center-star:** pick the sequence `c` minimising `Σ_b D[c][b]`.
- **Stage 3 — progressive merge:** NW-align each sequence to the center, then
  merge all those alignments in the center's coordinate frame ("once a gap, always
  a gap"). Grade with the **Sum-of-Pairs** column score.

See [THEORY.md](THEORY.md) for the full derivation, the merge worked example, and
where this sits relative to neighbor-joining / iterative-refinement MSA.

## Build

Requires **Visual Studio 2026** (v145) + **CUDA 13.3** ([docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/multiple-sequence-alignment-msa.sln`.
2. Select **`Release|x64`** → **Build Solution** →
   `build/x64/Release/multiple-sequence-alignment-msa.exe`.

CLI: `msbuild build\multiple-sequence-alignment-msa.sln /p:Configuration=Release /p:Platform=x64`

## Run the demo

```powershell
./demo/run_demo.ps1
```

Builds if needed, aligns the committed synthetic family, prints the multiple
alignment + Sum-of-Pairs score, and verifies the GPU pairwise-score matrix equals
the CPU matrix exactly.

## Data

- **Sample (committed):** `data/sample/sequences_sample.fasta` — **6 synthetic**
  DNA sequences descended from one ancestral motif (point mutations + small
  indels), so a correct MSA visibly lines up the conserved core.
- **Full data:** real MSA benchmarks (BAliBASE, HomFam, Pfam seeds) — see
  `scripts/download_data.ps1` / `.sh` and [data/README.md](data/README.md).
- Regenerate / enlarge: `python scripts/make_synthetic.py --n 12 --sub 0.10`.

## Expected output

`demo/expected_output.txt` holds the deterministic stdout (sequence count, chosen
center, the aligned rows, the conservation line, the Sum-of-Pairs score). The GPU
kernel (`src/kernels.cu`) and the serial CPU reference (`src/reference_cpu.cpp`)
compute the **same integer score matrix** via the shared recurrence in
`src/nw_core.h`, so they agree **exactly** (`mismatches = 0`, tolerance `0`).

## Code tour

1. [`src/nw_core.h`](src/nw_core.h) — the shared `__host__ __device__` NW score
   recurrence (CPU and GPU run *this* same math).
2. [`src/main.cu`](src/main.cu) — load, run CPU + GPU stage 1, compare matrices,
   assemble, print.
3. [`src/kernels.cuh`](src/kernels.cuh) — the one-block-per-pair interface + the
   pattern diagram.
4. [`src/kernels.cu`](src/kernels.cu) — the pairwise-scoring kernel + the host
   distance-matrix wrapper.
5. [`src/reference_cpu.h`](src/reference_cpu.h) / [`src/reference_cpu.cpp`](src/reference_cpu.cpp)
   — data model, FASTA loader, serial distance matrix, center-star + progressive
   assembly, Sum-of-Pairs.

## Prior art & further reading

- **MAFFT** (<https://mafft.cbrc.jp/alignment/software/>) — fastest large-scale CPU
  MSA; its PartTree distance phase has a GPU-accelerated prototype (the speedup we
  cite). Study its progressive + iterative strategy.
- **CUK-Band (2024)** (<https://link.springer.com/chapter/10.1007/978-981-97-5692-6_8>)
  — CUDA **center-star** MSA with banded DP; the academic basis for this project's
  guide choice.
- **MMseqs2-GPU** (<https://github.com/soedinglab/MMseqs2>) — GPU-accelerated MSA
  *search* that builds the deep MSAs feeding AlphaFold2.
- **Clustal Omega / ClustalW** (<http://www.clustal.org/>) — the canonical
  progressive-MSA references (guide trees, profile-profile alignment).

Study these for the production approach; reimplement didactically (CLAUDE.md §2).

## CUDA pattern used here

**Independent jobs · one thread block per pair** (the catalog's distance-matrix
pattern) · shared `__host__ __device__` NW recurrence for exact CPU/GPU parity ·
per-pair DP rows in **shared memory** · integer scoring (exact, deterministic
verification). Contrast `3.01` (parallelism *within* one alignment) and `1.12`
(one *thread* per independent job).

## Exercises

1. **Wavefront-within-pair.** Replace the single-thread serial DP in each block
   with the anti-diagonal wavefront of `3.01`, using all 32 lanes. Measure the win
   on longer sequences.
2. **Neighbor-joining guide.** Swap center-star for an NJ guide tree built from the
   distance matrix; align along the tree instead of to a single center.
3. **Affine gaps.** Move from a linear gap penalty to gap-open/gap-extend (the
   Gotoh recurrence) in `nw_core.h`; the CPU/GPU parity stays exact.
4. **Iterative refinement.** After the first MSA, remove one sequence, re-align it
   to the profile of the rest, and keep the change if the Sum-of-Pairs score
   improves. Repeat.
5. **Bigger families.** Run `make_synthetic.py --n 64`; watch the pair count grow
   `O(N²)` and the GPU's relative advantage with it.

## Limitations & honesty

- **Reduced-scope teaching version.** Real MSA (Active R&D in the catalog) uses
  guide *trees* (NJ), profile-profile alignment, affine gaps, and iterative
  refinement. We implement the simplest correct progressive variant (center-star,
  linear gaps) and describe the fuller pipeline in THEORY.
- **Center-star merge** does not mutually align *insertions relative to the
  center*: residues that sit in the same merged gap column come from different
  sequences and are co-located, not truly aligned (a known property of star
  alignment — THEORY §algorithm).
- **DNA + linear gaps only**; protein MSA needs a substitution matrix (BLOSUM/PAM).
- **Per-pair launches on tiny N** make the GPU *slower* than the CPU here (launch
  overhead) — an honest teaching artifact; the GPU wins as N and L grow.
- The within-pair DP is serial in this version (the across-pairs parallelism is
  the point). Production tools parallelise both levels.
