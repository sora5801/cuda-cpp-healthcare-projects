# 3.4 — Nanopore Basecalling

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Genomics%2C%20Sequencing%20%26%20Bioinformatics-lightgrey)

> **🟢 Beginner · Established** — Domain 3: Genomics, Sequencing & Bioinformatics · Catalog ID `3.4`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

> **⚠️ Reduced-scope teaching version (CLAUDE.md §13).** Full nanopore basecalling
> is a *trained neural network* (LSTM/transformer) followed by a *CTC decoder*.
> The network is research-grade — trained weights, cuDNN/TensorRT — and is **out of
> scope** here. This project implements the **second stage: the CTC greedy decode**,
> the step that turns the network's per-timestep base probabilities into an actual
> DNA sequence. It is the part a learner can fully understand, parallelize, and
> verify exactly. The full pipeline is described in [THEORY.md](THEORY.md) under
> "Where this sits in the real world".

## Summary

A nanopore sequencer measures the tiny ionic current as a DNA strand ratchets
through a protein pore; a neural network turns that "squiggle" into, for each
read, a matrix of probabilities — at every time step, how likely each of
`{blank, A, C, G, T}` is. **This project takes that probability matrix and decodes
it into bases** using *greedy CTC collapse* (argmax each step, merge repeats, drop
blanks). Every read decodes independently, so we hand each read to its own GPU
thread — the simplest and most fundamental GPU parallel pattern. We run the exact
same decoder on the CPU and the GPU and check they agree **bit-for-bit**.

## What this computes & why the GPU helps

Nanopore basecalling translates raw ionic-current signal samples (electrical squiggles) from the sequencer into DNA/RNA base sequences. Oxford Nanopore's Dorado uses a recurrent neural network (transformer + CTC decoder in current "SUP" models) trained to map signal windows to base probabilities. The bottleneck is the RNN/transformer inference over millions of signal events per run hour, a perfect GPU workload: batched matrix multiplications across reads mapped to thousands of CUDA cores. Dorado achieves up to 30% speed improvement for HAC models on Ampere/Ada/Blackwell GPUs over previous versions and scales linearly across multiple GPUs. The GPU also powers modified base (methylation) calling simultaneously during basecalling.

**The parallel bottleneck (this project):** a sequencing run produces *millions of
reads*, and each read's posterior matrix must be decoded into bases. Those decodes
are completely independent of one another, so the work is *embarrassingly parallel
across reads*. We map **one read → one GPU thread**: thousands of reads decode
concurrently. (In production the dominant cost is the network inference itself —
batched GEMMs via cuBLAS/cuDNN/TensorRT — but that is the out-of-scope stage; see
THEORY.)

## The algorithm in brief

Real pipeline (full): Bidirectional LSTM / Transformer encoder; Connectionist
Transformation Classification (CTC) decoding; beam search decoding; adaptive banded
event alignment (f5c); modified-base (5mC, 6mA) classification heads.

**Implemented here (the decode stage):**

- **Argmax per time step** — pick the most probable class at each of `T` steps.
- **CTC collapse** — merge runs of the same class (a base dwells several steps →
  one call), then delete all blanks.
- **One read per GPU thread** — a grid-stride loop over the batch; no atomics, no
  shared memory, no races (each read's output is disjoint).
- **Exact verification** — the CPU and GPU call the *identical* `__host__ __device__`
  decode (`src/ctc_core.h`), so they agree exactly (tolerance `0`).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation, including why greedy decode is a special case of full CTC and what beam
search adds.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/nanopore-basecalling.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/nanopore-basecalling.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\nanopore-basecalling.sln /p:Configuration=Release /p:Platform=x64
```

This project links only the CUDA runtime (`cudart_static.lib`) — no extra GPU
libraries — so the build is dependency-light.

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/reads_sample.txt`, prints the
decoded sequence of each read, shows the GPU-vs-CPU agreement check, and prints a
timing line to stderr.

## Data

- **Sample (committed):** `data/sample/reads_sample.txt` — a tiny, offline batch of
  4 **synthetic** posterior matrices with *planted* DNA sequences, so the demo runs
  with zero downloads and we can confirm it recovers the known answer.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print how to obtain real ONT
  data (and `scripts/make_synthetic.py` regenerates the synthetic sample).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: ONT Open Dataset (PromethION human WGS) — available via SRA / ENA (https://www.ncbi.nlm.nih.gov/sra); R9.4.1 and R10.4.1 benchmark datasets released by ONT (https://github.com/GoekeLab/awesome-nanopore); GIAB ONT ultra-long reads — NA12878/HG002 nanopore truth sets (https://www.nist.gov/programs-projects/genome-bottle); ENA Project PRJNA594038 — public multi-species ONT data (https://www.ebi.ac.uk/ena).

> **Note on real data.** Production basecalling consumes raw signal (`.pod5`/`.fast5`)
> and *runs the network* to produce posteriors. This teaching project consumes the
> *posteriors directly* (the network's output), because the network is the
> out-of-scope stage. The download script explains this and points to real tools.

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt). The
program decodes each read on both the **GPU** (`src/kernels.cu`) and a **CPU
reference** (`src/reference_cpu.cpp`) and asserts they agree **exactly** — same
length, same base string, same checksum for every read. Because the synthetic
sample plants known sequences, you can read them straight off the output:

```
  read 0: T=32  len=8  checksum=6c2d4a63  seq=ACGTACGT
  read 1: T=32  len=8  checksum=faa0aca5  seq=AACCGGTT
  read 2: T=28  len=7  checksum=6fbc9be0  seq=GATTACA
  read 3: T=32  len=8  checksum=bb5f5785  seq=TTTTGGGG
...
RESULT: PASS (GPU matches CPU exactly; tol = 0, integer decode)
```

Read 1 (`AACCGGTT`) and read 2's `TT` and read 3 (`TTTTGGGG`) are *homopolymers* —
the case where CTC's blank symbol matters — and the decoder recovers them intact.

## Code tour

Read in this order:

1. [`src/ctc_core.h`](src/ctc_core.h) — **start here.** The `__host__ __device__`
   shared core: the CTC alphabet, argmax, and the greedy collapse. CPU and GPU both
   call this, which is *why* they agree exactly.
2. [`src/main.cu`](src/main.cu) — loads the batch, runs CPU + GPU, verifies, reports.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the one-read-per-thread idea.
4. [`src/kernels.cu`](src/kernels.cu) — the kernel (grid-stride over reads) and host wrapper.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline + the data loader.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

Dorado (https://github.com/nanoporetech/dorado) — ONT's official GPU basecaller, multi-GPU, CUDA-optimised, supports MOD calling; f5c (https://github.com/hasindu2008/f5c) — CUDA-accelerated methylation calling and event alignment; awesome-nanopore (https://github.com/GoekeLab/awesome-nanopore) — curated tool index including GPU-enabled callers; Guppy — legacy ONT CUDA basecaller, GPU-only, superseded by Dorado.

One line each on what to learn from them:

- **Dorado** — the modern reference: see how the network output feeds a CUDA CTC
  decode and how reads are batched across GPUs. Our decode mirrors its greedy path.
- **f5c** — how event alignment / methylation calling is CUDA-accelerated; a good
  next step beyond plain basecalling.
- **awesome-nanopore** — a curated index for finding datasets and GPU-enabled tools.
- **Guppy (legacy)** — Dorado's predecessor; useful for understanding the historical
  CTC "fast"/"hac"/"sup" model tiers.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Independent jobs · one thread per item** (PATTERNS.md §1, exemplified by `1.12`
Tanimoto). Each read is an independent decode, so thread
`i = blockIdx.x*blockDim.x + threadIdx.x` decodes read `i`, with a **grid-stride
loop** so one modest grid covers a batch of any size. There are **no atomics and no
shared memory** — each read writes its own output row — which is exactly why this
parallelizes perfectly. The decode math is integer-only and lives in a
`__host__ __device__` core, so CPU and GPU produce **bit-identical** results
(PATTERNS.md §2, §3). (The full production pipeline additionally uses cuBLAS/cuDNN/
TensorRT for the network's GEMMs and CUDA streams to pipeline signal batches; those
belong to the out-of-scope stage.)

## Exercises

1. **Beam search.** Greedy decode keeps only the single best path. Implement
   prefix-beam-search CTC (top-`k` hypotheses) on the CPU and compare calls on a read
   where two classes are nearly tied at some steps. Where does greedy go wrong?
2. **Quality scores.** Emit a Phred-style per-base quality from the posterior
   probability of the called class (`Q = -10·log10(1 - p)`), and add it to the output.
3. **Homopolymer stress test.** Edit `make_synthetic.py` to plant a long run like
   `AAAAAAAA` and shrink the blank separators; observe how greedy CTC under-calls
   homopolymers — the classic nanopore error mode. Document what you see in THEORY terms.
4. **One block per read, parallel argmax.** For very long reads, give each read a
   *block* and parallelize the argmax across threads with a shared-memory reduction.
   Measure whether it helps at the sample's read length (it likely will not — explain why).
5. **FASTA output.** Write the decoded reads to a `.fasta` file and align them to the
   planted truth with project `3.01` (Smith-Waterman) to compute percent identity.

## Limitations & honesty

- **Reduced scope.** The neural network — the hard, research-grade part — is **not**
  implemented. We decode *given* posteriors. This is a deliberate teaching slice
  (CLAUDE.md §13), not a basecaller you can point at a sequencer.
- **Synthetic data.** The committed sample is **synthetic**: we hand-build posterior
  matrices that decode to planted sequences. They are *not* real network outputs and
  the probabilities are idealized so the argmax is unambiguous. Labeled synthetic
  everywhere.
- **Greedy only.** Real "sup" models use richer decoders (CTC beam search, sometimes
  conditional random fields). Greedy is the simplest correct decode and a strict
  subset; it under-calls homopolymers and ignores near-ties.
- **Timing is a teaching artifact**, never a benchmark (CLAUDE.md §12). On this tiny
  batch the GPU is launch-bound and *slower* than the CPU; its advantage appears at
  run scale (millions of reads). No clinical claims — educational only.
