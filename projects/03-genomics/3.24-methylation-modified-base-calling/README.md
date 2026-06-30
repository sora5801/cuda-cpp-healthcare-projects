# 3.24 — Methylation / Modified-Base Calling

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Genomics%2C%20Sequencing%20%26%20Bioinformatics-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 3: Genomics, Sequencing & Bioinformatics · Catalog ID `3.24`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

A nanopore sequencer measures the ionic current as a DNA strand threads through a
protein pore; the current depends on the few bases ("k-mer") sitting in the pore.
When a cytosine is **methylated** (5-methylcytosine, **5mC**) it shifts that
current slightly. This project calls methylation the way **f5c** does: for every
(read, CpG-site) pair it runs a small **banded event-alignment dynamic program**
that threads the read's observed current events onto the reference k-mers, scores
that alignment under **two competing pore models** (canonical C vs. 5mC), and
takes the **log-likelihood ratio (LLR)** of the two. Average the LLRs of all reads
covering a site and you get a per-site methylation call. The thousands of per-pair
alignments are independent, so each becomes one GPU thread — the classic batched
independent-jobs pattern.

This is a **reduced-scope teaching version** (CLAUDE.md §13): a 3-mer pore model,
a fixed band, and synthetic signal — small enough to read by eye, faithful enough
to teach the real algorithm. It is **not** a basecaller and does no neural-network
inference (that is Remora/Dorado territory; see *Limitations*).

## What this computes & why the GPU helps

Detection of DNA methylation (5mC, 5hmC) and other modifications (6mA, BrdU) from
nanopore raw signal requires classifying the ionic-current waveform at each
potentially modified site. f5c's GPU-accelerated **adaptive banded event
alignment** assigns signal events to reference positions using GPU-parallelised
DP, then scores modification probability. ONT Remora trains small CNN/LSTM models
to classify modifications directly from raw signals, with GPU inference integrated
into Dorado basecalling. Accurate genome-wide 5mCG calling at 30× ONT coverage
processes **billions** of signal samples.

**The parallel bottleneck:** the per-site work is one banded DP per read per pore
model. A 30× human methylome is ~28 million CpG sites × ~30 reads × 2 models ≈
**1.7 billion small DPs**. Every one is independent of the others, so the work is
embarrassingly parallel: assign **one GPU thread per (read, site) job** and the
whole batch runs at once. The DP itself is tiny (a banded recurrence over a short
window), so the win is throughput across millions of jobs, not speed of any single
DP. This is exactly the shape of flagship `1.12` (Tanimoto) and `12.01` (spectral
search): score many independent items against a shared, constant-memory reference.

## The algorithm in brief

- **Pore model.** A trained table mapping each k-mer → expected current (Gaussian
  mean + stdv). We keep **two**: canonical and methylated. They differ only on the
  k-mers that contain the CpG cytosine.
- **Banded event-alignment DP.** A Needleman–Wunsch-style recurrence aligns the
  read's events to the reference k-mers with `match` / `stay` (extra event on the
  same k-mer) / `skip` moves, restricted to a diagonal band. The per-cell cost is
  the Gaussian emission log-probability of the event under the aligned k-mer. We
  take the best path's **log-likelihood** (Viterbi / max-product in log space).
- **Log-likelihood-ratio scoring.** `LLR = logL(methylated) − logL(canonical)`
  per (read, site). Positive ⇒ the events look more like 5mC.
- **Per-site call.** Average the LLRs over the site's reads; call 5mC if the mean
  LLR > 0 (a deterministic integer decision).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation, including f5c's *adaptive* band (which we simplify to a fixed band).

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/methylation-modified-base-calling.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/methylation-modified-base-calling.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\methylation-modified-base-calling.sln /p:Configuration=Release /p:Platform=x64
```

This project links only `cudart_static.lib` (no extra CUDA libraries): the banded
DP and the Gaussian emission are hand-rolled so nothing is a black box.

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/methylation_sample.txt`, prints
the per-site calls, shows the GPU-vs-CPU agreement check, and prints a timing line
to stderr.

## Data

- **Sample (committed):** `data/sample/methylation_sample.txt` — a tiny, **synthetic**
  instance (12 CpG sites, 8 reads each) so the demo runs with zero downloads. It
  embeds a known ground truth so the program can report calling accuracy.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print pointers to real
  ONT/ENCODE methylation data (idempotent; they never bypass credentials).
- **Bigger synthetic instance:** `python scripts/make_synthetic.py --sites 4096`.
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: ENCODE WGBS — genome-wide bisulfite methylation reference
(https://www.encodeproject.org/); Oxford Nanopore open datasets — R10.4.1 with
5mC/6mA labels (https://github.com/GoekeLab/awesome-nanopore); NCBI GEO methylation
studies (https://www.ncbi.nlm.nih.gov/geo/).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt): a table
of 12 per-site calls, `calls matching ground truth: 12 of 12`, and
`RESULT: PASS`. The program computes the per-job LLRs on both the **GPU**
(`src/kernels.cu`) and a **CPU reference** (`src/reference_cpu.cpp`) and asserts
they agree within tolerance `1.0e-3` — that agreement is the correctness guarantee.
Because both sides call the *same* `__host__ __device__` DP core
([`src/meth_core.h`](src/meth_core.h)), the measured `max_abs_err` is `0.0` on this
machine.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads data, runs CPU + GPU, verifies, reports.
2. [`src/meth_core.h`](src/meth_core.h) — the shared physics: Gaussian emission +
   the banded DP, written once as `__host__ __device__` so CPU and GPU match.
3. [`src/reference_cpu.h`](src/reference_cpu.h) / [`.cpp`](src/reference_cpu.cpp) —
   the data model and the trusted serial baseline.
4. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the thread-mapping idea.
5. [`src/kernels.cu`](src/kernels.cu) — the kernel (one thread per job) + host wrapper.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

- **f5c** (https://github.com/hasindu2008/f5c) — CUDA-accelerated methylation
  calling and event alignment. **Study:** its adaptive banded DP and the pore-model
  scoring; this project is a didactic miniature of exactly that.
- **Remora** (https://github.com/nanoporetech/remora) — ONT modified-base model
  training/calling. **Study:** how a learned CNN replaces the hand-built pore model.
- **Dorado** (https://github.com/nanoporetech/dorado) — integrates modification
  calling into basecalling on GPU. **Study:** the streaming multi-read pipeline.
- **Modkit** (https://github.com/nanoporetech/modkit) — downstream modified-base
  analysis (bedMethyl). **Study:** how per-site calls aggregate into a methylome.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Batched independent jobs · constant-memory shared model** (PATTERNS.md §1, the
`1.12`/`12.01` row). One thread per (read, site) job; the two pore models live in
`__constant__` memory (read by every thread, never written during the launch, so
the constant cache broadcasts each entry warp-wide); the per-job DP scratch (two
short rows of doubles) lives in registers/local memory. No atomics, no shared
memory, no inter-thread communication ⇒ deterministic and embarrassingly parallel.
The per-element physics is shared with the CPU via a `__host__ __device__` core
(PATTERNS.md §2), making verification exact.

## Exercises

1. **Adaptive band.** Replace the fixed band with f5c's adaptive band that
   re-centers on the running best cell each row. Does the call accuracy change on a
   noisier instance (`--jitter 4.0`)?
2. **Per-read coverage.** Lower `--coverage` to 2 and raise `--jitter`. At what
   point does the mean-LLR sign start flipping for true sites? (This is the
   coverage-vs-confidence trade-off real methylomes face.)
3. **Shared memory.** Stage the two pore-model rows a block needs into shared
   memory and compare occupancy/timing for a large `--sites`. (At k=9 the model no
   longer fits constant memory — see THEORY "real world".)
4. **FP32 DP.** Run the DP emission in `float` instead of `double` and watch
   `max_abs_err` grow; decide an honest tolerance for that precision (PATTERNS §4).
5. **5hmC third model.** Add a third pore model (5-hydroxymethylcytosine) and turn
   the binary call into a 3-way argmax over canonical / 5mC / 5hmC.

## Limitations & honesty

- **Synthetic data.** The signal, pore models, and ground truth are generated by
  `scripts/make_synthetic.py` and labeled synthetic everywhere. The clean
  separation of LLRs here is *easier* than real nanopore data.
- **Reduced scope.** We use a **3-mer** pore model (real R10 models are 9-mers,
  262 144 k-mers), a **fixed** band (f5c's is adaptive), a **1:1** event-to-k-mer
  layout (real segmentation yields variable event counts), and **no basecalling**
  (we assume the reference window is known). Each simplification is explained in
  THEORY.md and left as an exercise.
- **No neural network.** Modern callers (Remora/Dorado) classify with trained
  CNN/LSTM models, not a fixed Gaussian pore model. This project teaches the
  classical f5c-style likelihood path, which remains the clearest way to *see* why
  a current shift implies methylation.
- **Not clinical.** Nothing here may inform diagnosis or treatment.
