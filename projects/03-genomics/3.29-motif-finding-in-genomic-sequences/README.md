# 3.29 — Motif Finding in Genomic Sequences

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Genomics%2C%20Sequencing%20%26%20Bioinformatics-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 3: Genomics, Sequencing & Bioinformatics · Catalog ID `3.29`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

Transcription factors bind short DNA patterns (~6–20 bp) called **motifs**. Given
a set of DNA sequences that all bind the same factor — e.g. the regions under
ChIP-seq peaks — *motif finding* recovers that shared, over-represented pattern
even though every copy is slightly different and sits at an unknown position.
This project implements the classic **MEME** approach: model the motif as a
**position weight matrix (PWM)** and fit it by **Expectation-Maximisation (EM)**.
The expensive inner step — scoring every length-W window of every sequence
against the current PWM — is offloaded to a CUDA kernel (one thread per window).
On a tiny synthetic sample with a planted motif, the program recovers the motif
from scratch and reports the predicted binding site in each sequence.

## What this computes & why the GPU helps

Transcription factor motif discovery from ChIP-seq peaks searches for
over-represented sequence patterns (PWMs) against a background model. EM over all
N×W sequence windows (N peaks × ~W-k+1 positions per peak) is the cost driver;
GPU parallelism assigns one thread to each window position, computing the PWM
score via a parallel dot product. mCUDA-MEME achieves orders-of-magnitude speedup
by distributing MEME's EM steps across GPU cores. For genome-scale ChIP-seq
(millions of peaks), this turns multi-day CPU runs into hours.

**The parallel bottleneck:** the **E-step window scoring**. Each EM iteration must
score *every* length-W window of *every* sequence — for total sequence length L
that is ~L independent PWM dot products, each a W-term sum over a small log-odds
table. This dominates the runtime, and every window is independent — a textbook
"many independent jobs" workload. We give each window its own GPU thread; the
cheap EM bookkeeping (softmax, count accumulation) stays on the host.

## The algorithm in brief

- **Model:** a W×4 PWM (per-column base probabilities) + a background base
  composition. Their ratio gives a **log-odds** table.
- **E-step (GPU):** score every window = sum of W log-odds lookups; per sequence,
  a numerically-stable softmax turns scores into **responsibilities** (posterior
  over the motif's offset, OOPS model).
- **M-step (CPU):** re-estimate the PWM as responsibility-weighted base counts
  with a pseudocount; renormalise.
- **Iterate** until the data log-likelihood stops changing; report the consensus
  (argmax per column), information content, and per-sequence best site.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/motif-finding-in-genomic-sequences.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/motif-finding-in-genomic-sequences.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\motif-finding-in-genomic-sequences.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/sequences_sample.fasta`, prints
the recovered motif + predicted sites, shows the GPU-vs-CPU agreement check, and
prints a timing line.

## Data

- **Sample (committed):** `data/sample/sequences_sample.fasta` — 12 synthetic
  60 bp sequences, each with a planted (mutated) copy of the motif `TGACGTCA`.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print how to assemble a
  ChIP-seq peak FASTA (peak BED + `bedtools getfasta`).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: ENCODE ChIP-seq peak BED files (<https://www.encodeproject.org/>);
JASPAR 2024 curated PWM database (<https://jaspar.elixir.no/>); ReMap 2022
(<https://remap.univ-amu.fr/>); GEO ChIP-seq datasets
(<https://www.ncbi.nlm.nih.gov/geo/>).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt): EM
converges, the recovered consensus is `CTTGACGT` (the planted `TGACGTCA` core
captured one register left — a genuine EM phase shift, explained in the demo
README and THEORY), information content ≈ 8.27 bits, and per-sequence sites are
printed. The program computes the E-step on both the **GPU** (`src/kernels.cu`)
and a **CPU reference** (`src/reference_cpu.cpp`) and asserts they agree
**exactly** (tolerance `0`) — that agreement is the correctness guarantee.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads sequences, runs CPU EM, runs the GPU
   E-step on the final model, verifies, reports.
2. [`src/motif_core.h`](src/motif_core.h) — the shared `__host__ __device__`
   `window_score()`: the one formula CPU and GPU both call (parity idiom).
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the thread-mapping idea.
4. [`src/kernels.cu`](src/kernels.cu) — the window-scoring kernel (constant-memory
   log-odds table) and host wrapper.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the FASTA loader and the
   trusted serial MEME EM driver.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

- **MEME Suite** (<https://meme-suite.org/>) — the reference CPU motif toolkit;
  this project reimplements its OOPS EM core didactically. Study its ZOOPS/TCM
  variants and the E-value statistics we omit.
- **CUDA-MEME / mCUDA-MEME** (<https://cuda-meme.sourceforge.io/homepage.htm>) —
  GPU-accelerated MEME; learn how it batches the EM steps across GPU cores and
  multiple GPUs (the production version of what we do for one E-step).
- **Argo_CUDA** (<https://pubmed.ncbi.nlm.nih.gov/29281953/>) — exhaustive GPU
  motif discovery for large datasets; a different (enumeration) GPU strategy.
- **HOMER** (<http://homer.ucsd.edu/>) — CPU ChIP-seq motif enrichment; study its
  known-motif vs de-novo enrichment testing.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## Exercises

1. **Sweep the motif width.** The code fixes `W=8`. Run EM for several widths and
   keep the one with the highest information content — that is how real tools
   choose W.
2. **Fix the phase shift.** EM lands on `CTTGACGT` (shifted left by 2). Add a
   final "shift refinement" pass that tries shifting the recovered PWM ±k columns
   and keeps the best — does it snap back to `TGACGTCA`?
3. **Multiple random restarts.** EM finds a *local* optimum. Seed several random
   PWMs, run each to convergence, and keep the best by likelihood (and report how
   often each restart finds the true motif).
4. **Search both strands.** TF motifs occur on either DNA strand. Also score the
   reverse complement of each window and take the better orientation.
5. **Profile the kernel.** Increase the synthetic input (`--n 5000 --len 200`) and
   compare GPU vs CPU E-step time as L grows — when does the GPU pull ahead?

## Limitations & honesty

- **Reduced-scope teaching version.** We implement the **OOPS** model (exactly one
  motif occurrence per sequence) for a **single, fixed width**, with the M-step on
  the CPU. Production MEME adds the ZOOPS/TCM occurrence models, automatic width
  selection, statistical significance (E-values), erasing found motifs to find
  more, and both-strand search. Those are described in THEORY "real world" and
  left as exercises.
- **Synthetic data.** The committed sample is synthetic, with a planted motif and
  10% mutation — labeled synthetic everywhere. The recovered motif and sites are
  meaningful only for this toy input; **no biological or clinical conclusion** may
  be drawn.
- **The GPU only accelerates one step.** This is intentional and honest: it
  isolates the parallelisable bottleneck (E-step scoring) so the speed-up is
  legible. On the tiny sample the GPU is launch-bound and *slower* than the CPU;
  its advantage appears at ChIP-seq scale (millions of windows).
- **Local optimum / phase shift.** EM converges to a local optimum; here that is a
  2 bp-shifted register of the true motif. This is real EM behaviour, reported
  honestly rather than hidden.
