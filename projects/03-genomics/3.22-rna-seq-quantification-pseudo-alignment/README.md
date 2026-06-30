# 3.22 — RNA-seq Quantification / Pseudo-alignment

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Genomics%2C%20Sequencing%20%26%20Bioinformatics-lightgrey)

> **🟢 Beginner · Established** — Domain 3: Genomics, Sequencing & Bioinformatics · Catalog ID `3.22`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

RNA-seq tells you *how much* of each transcript a cell is expressing — but a
single sequencing read often matches several isoforms of the same gene, so you
cannot just count reads per transcript. **Pseudo-alignment** (kallisto, Salmon)
solves this fast: instead of aligning every read base-by-base, it hashes each
read's k-mers, looks up which transcripts are *compatible*, and groups reads that
hit the same transcript set into an **equivalence class**. The abundances are then
recovered by an **expectation-maximisation (EM)** loop that statistically divides
each ambiguous class among its members. This project implements that EM on the
GPU: one thread per equivalence class runs the E-step, and the M-step is a
deterministic fixed-point atomic reduction. On a tiny synthetic dataset with a
*known* answer, the EM recovers the true abundances to ~1e-5 and the GPU matches
the CPU reference exactly.

## What this computes & why the GPU helps

Pseudo-alignment (kallisto, Salmon) bypasses full read alignment by mapping k-mers directly to equivalence classes of transcripts, then running the EM algorithm to estimate transcript abundances. GPU acceleration of kallisto redesigns the k-mer compatibility look-up and EM optimisation for GPU throughput: k-mer hash table queries map naturally to parallel GPU hash probes, and the EM update over millions of reads is a dense GEMV. A 2026 study ("RNA-seq analysis in seconds using GPUs," Melsted et al.) demonstrates GPU kallisto completing quantification in seconds vs. minutes on CPU. Salmon's variational Bayes EM is similarly GPU-amenable.

**The parallel bottleneck:** the **EM iteration**. Real runs have 10^5–10^7
equivalence classes and tens of EM iterations; each iteration re-reads every
class, re-estimates how its reads split among the current abundances, and
accumulates the result — a sparse matrix-vector sweep that dominates runtime. The
classes are independent, so we give each its own GPU thread (the E-step), and the
accumulation into per-transcript totals is an atomic reduction (the M-step). This
is the "EM update over millions of reads is a [sparse] GEMV" the deep dive names.
(We hand-roll the sweep so nothing is a black box; THEORY.md explains where
cuSPARSE would slot in.)

## The algorithm in brief

K-mer de Bruijn graph construction for transcriptome index; pseudoalignment compatibility class assignment; expectation-maximisation (EM) for abundance estimation; variational Bayes EM (Salmon); bootstrap resampling for uncertainty; quasi-mapping hash-based alignment.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/rna-seq-quantification-pseudo-alignment.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/rna-seq-quantification-pseudo-alignment.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\rna-seq-quantification-pseudo-alignment.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/`, prints the result, shows the
GPU-vs-CPU agreement check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/` — a tiny, offline input so the demo runs
  with zero downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` (documented, idempotent).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: GENCODE human transcriptome — reference transcript index (https://www.gencodegenes.org/); ENCODE RNA-seq FASTQs — diverse cell-type transcriptomes (https://www.encodeproject.org/); GTEx v9 — tissue RNA-seq compendium (https://gtexportal.org/); SRA RNA-seq studies (https://www.ncbi.nlm.nih.gov/sra).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt): a
per-transcript table of `est_counts`, `rho`, `TPM`, and the embedded `truth_rho`,
followed by `recovery: L1(...) = 0.0000` and `RESULT: PASS`. The program runs the
EM on both the **GPU** (`src/kernels.cu`) and a **CPU reference**
(`src/reference_cpu.cpp`) and asserts their abundances agree within `1e-12` — in
fact they agree *exactly*, because the M-step reduction accumulates in commuting
fixed-point integers (see THEORY.md "How we verify correctness"). Two checks are
reported: GPU-vs-CPU agreement (the implementation is correct) and recovery of the
synthetic ground truth (the science is correct).

## Code tour

Read in this order:

1. [`src/pseudoalign.h`](src/pseudoalign.h) — the per-ec E-step math, written
   once as `__host__ __device__` so CPU and GPU run identical arithmetic. Start
   here; it is the conceptual core.
2. [`src/main.cu`](src/main.cu) — loads data, runs CPU + GPU EM, verifies, reports.
3. [`src/reference_cpu.h`](src/reference_cpu.h) / [`reference_cpu.cpp`](src/reference_cpu.cpp)
   — the `EcDataset`, the loader, the shared `counts_to_rho` renormalise, and the
   trusted serial EM baseline.
4. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the thread-mapping idea.
5. [`src/kernels.cu`](src/kernels.cu) — the EM iteration kernel and host loop.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

kallisto GPU branch (https://github.com/pachterlab/kallisto) — GPU branch for pseudo-alignment; Salmon (https://github.com/COMBINE-lab/salmon) — quasi-mapping quantification (GPU EM target); bustools (https://github.com/BUStools/bustools) — BUS file manipulation for scRNA-seq downstream; alevin-fry (https://github.com/COMBINE-lab/alevin-fry) — fast single-cell quantification, GPU-amenable.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Parallel per-item E-step + atomic fixed-point M-step reduction** (the same
family as flagship `11.09` k-means). One GPU thread owns one equivalence class:
it runs the E-step (split the class's reads among its member transcripts in
proportion to their length-normalised abundance) and then `atomicAdd`s each
member's expected reads into a per-transcript accumulator. The adds collide
(popular isoforms appear in many classes), so we accumulate in **fixed-point
integers** — integer adds commute, making the reduction both reproducible and
bit-identical to the CPU. The tiny renormalise that finishes each M-step runs on
the host and is shared with the CPU reference. The catalog also lists a GPU hash
table for k-mer lookup, cuSPARSE for the sparse class matrix, and CUDA streams for
I/O overlap; this teaching version focuses on the EM (the part that dominates
runtime) and discusses the rest in [THEORY.md](THEORY.md) "Where this sits in the
real world".

## Exercises

1. **Scale it.** Regenerate a much larger problem with
   `python scripts/make_synthetic.py --reads 5000000`, rerun, and watch the
   per-iteration GPU time. Where does the GPU start to win over the CPU? (Note our
   sample still has only 9 ecs — to truly stress the GPU you would also need many
   *more* ecs; try editing `SHARED_ECS` and the transcript set.)
2. **Keep `rho` on the device.** The current loop copies `rho` up and the counts
   down every iteration. Add a small GPU reduction kernel that normalises the
   counts into the next `rho` on-device, so the only host↔device traffic is the
   final answer. Measure the speed-up.
3. **Add a convergence stop.** Replace the fixed 100 iterations with "stop when the
   L1 change in `rho` < 1e-6". Why does the demo *not* do this by default? (Hint:
   determinism of stdout — see `docs/PATTERNS.md` §3.)
4. **Bootstrap uncertainty.** kallisto reports confidence intervals by resampling
   the reads. Run the EM on several resampled count vectors and report the spread
   of each transcript's TPM — an ensemble pattern (see flagship `9.02`).
5. **Block-size sweep.** Try `THREADS_PER_BLOCK` ∈ {32, 64, 128, 256, 512} and note
   that it barely matters here — explain why (the kernel is latency-bound on a tiny,
   irregular gather, not compute-bound).

## Limitations & honesty

- **The hard part is faked — on purpose.** Real pseudo-alignment *builds* the
  equivalence classes by k-mer hashing against a de Bruijn transcriptome index;
  this project starts from ecs already produced (we synthesize them) and implements
  only the EM that turns them into abundances. That is the deliberate teaching
  scope; the index/lookup is described in THEORY.md but not built here.
- **The data is synthetic.** Counts are generated by exactly the model the EM
  inverts, so recovery is unrealistically clean (~1e-5). Real data has mapping
  noise, sequencing error, multi-gene ambiguity, and GC/positional bias that
  kallisto/Salmon model and we do not — real EM converges to a *good* estimate, not
  the exact truth.
- **No bias correction, no bootstrap, single precision of effort.** We use FP64
  throughout for clean CPU/GPU parity but model none of Salmon's sequence-bias or
  variational-Bayes refinements.
- **Not a clinical tool.** This is study material. Nothing here is validated for
  any diagnostic or research-grade use.
