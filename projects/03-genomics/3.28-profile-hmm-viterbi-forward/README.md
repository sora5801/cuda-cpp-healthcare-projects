# 3.28 — Profile HMM (Viterbi / Forward)

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Genomics%2C%20Sequencing%20%26%20Bioinformatics-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 3: Genomics, Sequencing & Bioinformatics · Catalog ID `3.28`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

A **profile hidden Markov model (profile HMM)** turns a protein family into a
position-by-position scoring model: each column "knows" which amino acids belong
there. Tools like **HMMER** use profile HMMs to scan huge sequence databases and
find distant family members. This project builds a small, fully-commented profile
HMM and scores a database of sequences against it on the GPU using the two
classic dynamic-programming recurrences — **Viterbi** (the single best alignment)
and **Forward** (the total probability summed over all alignments). Each database
sequence is scored by its own GPU thread, mirroring how production GPU pHMM
search (e.g. CUDAMPF) parallelises across the database. The committed demo plants
a known homolog among random decoys and shows it ranking #1 by a wide margin,
with the GPU result verified bit-for-bit against a CPU reference.

## What this computes & why the GPU helps

Profile HMMs (pHMMs) model protein families as position-specific probability
distributions; HMMER3 searches databases by applying a cascade: MSV/SSV
(Multi-Segment Viterbi) filter, P7Viterbi, and Forward-Backward scoring. MSV/SSV
alone consumes ~72% of runtime. CUDAMPF parallelises the MSV/Viterbi recurrence
across database sequences: each CUDA thread (here) or thread block (CUDAMPF)
processes one query-profile versus one database sequence, computing the
profile×sequence score lattice. For very deep database scans (>10⁹ sequences in
metagenomics), GPU pHMM search reduces days to hours.

**The parallel bottleneck:** the per-sequence **dynamic-programming recurrence**
over the profile×sequence grid. Scoring one sequence of length `L` against a
profile of `M` columns costs `O(L·M)` and is the inner loop of the whole search.
Crucially, the `N` database sequences are **mutually independent**, so the search
parallelises trivially across sequences: one GPU thread runs one sequence's full
DP. See [THEORY.md §4](THEORY.md) for the thread/block mapping.

## The algorithm in brief

- **Plan-7-style profile HMM** with per-column **Match / Insert / Delete** states
  (a teaching subset of HMMER's full Plan-7 architecture).
- **Viterbi** — a max-sum DP over the `M[i][k] / I[i][k] / D[i][k]` lattice in log
  space: the log-probability of the single most likely state path (best alignment).
- **Forward** — the *same* recurrence with `max` replaced by **log-sum-exp**: the
  log-probability summed over **all** state paths (total family support).
- **Hit ranking** — sequences ranked by Viterbi score; the planted homolog wins.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation, including the begin/end simplifications and what HMMER adds.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/profile-hmm-viterbi-forward.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/profile-hmm-viterbi-forward.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\profile-hmm-viterbi-forward.sln /p:Configuration=Release /p:Platform=x64
```

This project links only the CUDA runtime (`cudart_static.lib`) — the kernels are
hand-written, so there is **no extra library to install** (THEORY §4 explains why
we hand-roll the DP instead of leaning on a library).

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if the CMake build is used)
```

The demo builds if needed, runs on `data/sample/phmm_sample.fasta`, prints the
ranked hits, shows the GPU-vs-CPU agreement check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/phmm_sample.fasta` — a tiny, **synthetic**
  FASTA file (1 consensus + 1 planted homolog + 6 random decoys) so the demo runs
  **offline with zero downloads**.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print instructions and
  links for the real corpora (they do not bypass any registration).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: Pfam-A — ~20 k protein family profiles
(<https://www.ebi.ac.uk/interpro/download/>); UniRef50 — protein sequences for
database search (<https://www.uniprot.org/help/uniref>); Rfam — RNA family
profiles (<https://rfam.org/>); JGI metagenome proteins — environmental pHMM
targets (<https://genome.jgi.doe.gov/>).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt): a table
ranking the 7 database sequences by Viterbi score, with `homolog` at rank 1
(`Viterbi -31.85`, tens of nats above the best decoy) and a final
`RESULT: PASS`. The program computes both scores on the **GPU** (`src/kernels.cu`)
and a **CPU reference** (`src/reference_cpu.cpp`) and asserts they agree within
`1.0e-4` nats — in fact they agree to `0.0e+00` because both sides run the
*identical* log-space arithmetic (THEORY §6). The scientific check is the
ranking: a correct profile HMM must place the homolog far above random decoys.

## Code tour

Read in this order:

1. [`src/phmm.h`](src/phmm.h) — **start here**: the shared `__host__ __device__`
   recurrence core (the model struct, log-sum-exp vs max, the per-cell math). This
   one header is why CPU and GPU agree exactly.
2. [`src/main.cu`](src/main.cu) — loads the FASTA, builds the profile from the
   consensus, runs CPU + GPU for both algorithms, verifies, prints the ranking.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the one-thread-per-
   sequence idea.
4. [`src/kernels.cu`](src/kernels.cu) — the device DP and host wrapper; compare it
   line-by-line against the CPU twin.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial DP, the
   FASTA loader, and the model builder.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

- **HMMER3** (<https://github.com/EddyLab/hmmer>) — the reference CPU implementation
  of profile-HMM search. Study its Plan-7 architecture, the MSV/Viterbi/Forward
  cascade, and its SIMD-vectorised DP. We model a teaching subset of Plan-7.
- **CUDAMPF** (<https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-016-0946-4>)
  — multi-tiered CUDA HMMER acceleration; study its *one block per sequence*
  shared-memory MSV/Viterbi kernel and warp-level reductions. We use the simpler
  *one thread per sequence* mapping; THEORY §7 contrasts the two.
- **MMseqs2** (<https://github.com/soedinglab/MMseqs2>) — a faster alternative that
  prefilters with k-mers before profile scoring; study the speed/sensitivity
  trade-off of prefiltering.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Independent jobs · constant-memory query** (PATTERNS.md §1), exemplified by
flagship `1.12` (Tanimoto) and `12.01` (spectral search): one query (the profile,
held in **constant memory**) scored against `N` independent items (the database
sequences, one per GPU thread via a **grid-stride loop**). The per-item inner work
here is a small **dynamic-programming recurrence** kept in each thread's local
memory, and the per-cell math lives in a shared `__host__ __device__` core so the
CPU reference and GPU kernel run byte-identical arithmetic (PATTERNS.md §2).

## Exercises

1. **Traceback.** Viterbi here returns only the *score*. Store back-pointers and
   reconstruct the best **alignment** (which residues map to which match columns),
   the way HMMER reports a domain hit.
2. **E-values.** Convert raw scores to **bit scores** against a null model and fit
   the score distribution (Gumbel/exponential tail) to report **E-values** — the
   statistic that makes HMMER hits interpretable.
3. **One block per sequence.** Re-map the kernel so a whole *block* cooperates on
   one sequence's DP (each thread owns a strip of profile columns, syncing per
   anti-diagonal). This is CUDAMPF's approach; compare occupancy and speed.
4. **Forward-Backward.** Add the Backward pass and compute **posterior
   probabilities** `P(state | sequence)` per cell — the basis of posterior
   decoding and confidence-annotated alignments.
5. **Bigger model.** Raise `MAX_M`/`MAX_L`, load a real Pfam consensus, and watch
   how local-memory pressure per thread affects occupancy (profile with Nsight).

## Limitations & honesty

- **Synthetic data, labeled as such.** The committed sample is generated by
  `scripts/make_synthetic.py` (a hand-picked consensus, a mutated homolog, random
  decoys). It is **not** real biological data and implies no biological finding.
- **Reduced-scope teaching model.** We implement the per-column **M/I/D** states
  and the Viterbi/Forward recurrences, but **simplify the Plan-7 begin/end**: a
  path enters at match column 1 (with a silent delete chain to skip leading
  columns) and ends in the final match column. HMMER's full N/B/E/C/J flanking
  states (local/glocal alignment, multi-hit) are described in THEORY §7, not coded.
- **Emissions are a toy.** Match columns favor a single consensus residue rather
  than being estimated from a real multiple-sequence alignment with
  Dirichlet-mixture priors. The *recurrence* is faithful; the *parameters* are
  didactic.
- **No traceback / no E-values.** We report scores and a ranking, not alignments
  or statistical significance (left as exercises 1–2).
- **One thread per sequence** is simple and correct but, for very long sequences,
  uses more local memory per thread than CUDAMPF's cooperative block scheme.
- **Timing is a teaching artifact, never a benchmark claim** (CLAUDE.md §12). On
  this tiny database the GPU is launch/copy-bound and slower than the CPU; the GPU
  edge appears only at metagenomic scale (millions–billions of sequences).
