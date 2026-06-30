# 3.6 — k-mer Counting & Minimiser Sketching

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Genomics%2C%20Sequencing%20%26%20Bioinformatics-lightgrey)

> **🟢 Beginner · Established** — Domain 3: Genomics, Sequencing & Bioinformatics · Catalog ID `3.6`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

This project counts **k-mers** — every length-`k` substring of a set of DNA reads —
and builds **minimiser MinHash sketches** to estimate how similar two read sets
are. k-mer counting is the foundation of genome-size estimation, sequencing-error
detection, assembly, and metagenomics; minimiser sketching is the trick behind
fast "are these the same species?" distance tools like Mash. We implement both on
the GPU with the **parallel-insert + atomic-reduce** pattern (a hand-rolled device
hash table) and a **sliding-window minimum**, then verify the GPU result against a
plain-C++ reference **exactly**. The committed sample is tiny and synthetic, with a
planted motif and a controlled overlap so the output recovers a known answer.

## What this computes & why the GPU helps

k-mer counting determines the frequency of every length-k substring in a read set,
foundational to genome-size estimation, error detection, assembly, and
metagenomics. For a 30× human genome (~270 Gb of sequence, k=21), the table has
~4 billion distinct k-mers; efficient parallel hashing and atomic counting saturate
GPU memory bandwidth. Gerbil uses GPU-resident hash tables and achieves >10× speed
over Jellyfish. Minimiser sketching (selecting a canonical subset of k-mers per
window) reduces data by ~5× and enables the MinHash / HyperMinHash distance
computations used in species typing; all operations parallelise across reads with
one GPU thread per minimiser.

**The parallel bottleneck:** there is one k-mer per read position, so a 30× human
genome yields ~10¹¹ k-mers — each must be canonicalised, hashed, and tallied. Every
k-mer is **independent** until it hits the shared count table, so we give **one GPU
thread per k-mer position** and let them collide only at the atomic counter. The
work is memory-bandwidth-bound (stream the reads, scatter into a hash table), which
is exactly what a GPU's wide memory bus is built for. Minimiser selection is an
independent per-window minimum, equally parallel.

## The algorithm in brief

- **2-bit encoding** of each k-mer (`A=00,C=01,G=10,T=11`) into a 64-bit word.
- **Canonicalisation:** `min(kmer, reverseComplement(kmer))` so both strands count
  as one (the reverse complement is a bit-twiddle, not a string reversal).
- **Counting:** insert each canonical k-mer into a **device open-addressing hash
  table** (linear probing; claim a slot with `atomicCAS`, tally with `atomicAdd`).
  Integer counts make the atomics commute → deterministic, exact vs. CPU.
- **Minimiser extraction:** slide a window of `w` consecutive k-mers and emit the
  one with the smallest hash (a strong avalanche mix, so selection is unbiased).
- **MinHash sketch:** keep the `s` smallest distinct minimiser hashes ("bottom-s").
- **Jaccard estimate:** the fraction of the merged bottom-s hashes shared by both
  sketches estimates `J(A,B) = |A∩B| / |A∪B|` (the Mash distance core).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation, including the sort-then-reduce alternative and the warp-shuffle
window-min.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/k-mer-counting-minimiser-sketching.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/k-mer-counting-minimiser-sketching.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\k-mer-counting-minimiser-sketching.sln /p:Configuration=Release /p:Platform=x64
```

The project links only the CUDA runtime (`cudart_static.lib`); no extra CUDA
library is required (the hash table and window-min are hand-rolled so the atomic
and probing patterns stay visible — see THEORY for the Thrust/cuRAND variants).

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/kmer_sample.txt`, prints the
result (top k-mers, sketch sizes, Jaccard estimate), shows the GPU-vs-CPU
agreement check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/kmer_sample.txt` — a tiny, **synthetic**
  two-set DNA file so the demo runs with zero downloads. It plants a motif and a
  controlled A/B overlap so the result is interpretable (see
  [data/README.md](data/README.md)).
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print idempotent SRA
  Toolkit commands (NA12878, GenomeTrakr, GAGE); they never bypass anything.
- **Synthetic generator:** `scripts/make_synthetic.py` (deterministic; supports
  larger problems via `--genome/--reads/--readlen`).

Catalog dataset notes: Illumina WGS of NA12878 — human reference dataset (https://www.ncbi.nlm.nih.gov/sra/SRR622457); GAGE benchmark — multi-species short reads for assembly tools (http://gage.cbcb.umd.edu/); GenomeTrakr pathogen WGS — bacterial surveillance reads (https://www.ncbi.nlm.nih.gov/bioproject/PRJNA183844); Sequence Read Archive (SRA) — global repository (https://www.ncbi.nlm.nih.gov/sra).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt). The
program computes the **k-mer histogram**, the **minimiser sketches** of A and B,
and their **Jaccard estimate** on both the **GPU** (`src/kernels.cu`) and a **CPU
reference** (`src/reference_cpu.cpp`), and asserts they agree **exactly**
(tolerance 0). Why exact and not approximate? Both sides share the per-k-mer math
in `src/kmer.h` and accumulate **integer** counts, so there is no floating-point
order dependence — the GPU hash table reproduces the CPU `std::map` key-for-key
and count-for-count, and the sketches match hash-for-hash. The planted motif
`ACGTACGTACG` is the top k-mer (count = 7), confirming the pipeline recovers a
known signal.

## Code tour

Read in this order:

1. [`src/kmer.h`](src/kmer.h) — the shared `__host__ __device__` core: 2-bit
   encode, reverse complement, canonicalisation, hash. **Start here** — both the
   CPU and the GPU call exactly these functions.
2. [`src/main.cu`](src/main.cu) — loads the two read sets, runs CPU + GPU, verifies
   histogram + sketches + Jaccard, prints the deterministic report.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the two thread-mapping
   ideas (hash-table insert, window minimum).
4. [`src/kernels.cu`](src/kernels.cu) — the device hash table (`atomicCAS` +
   `atomicAdd`) and the minimiser kernel, plus their host wrappers.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline
   and the sample loader.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, CUDA-event timer, I/O helpers.

## Prior art & further reading

- **Gerbil** (<https://github.com/uni-halle/gerbil>) — GPU-supported k-mer counter;
  study its GPU-resident hash table and how it spills to disk for huge inputs.
- **KMC3** (<https://github.com/refresh-bio/KMC>) — disk-I/O-efficient CPU counter;
  the baseline GPU counters are benchmarked against. Learn its (k, signature)
  bucketing.
- **Jellyfish** (<https://github.com/gmarcais/Jellyfish>) — a lock-free hash table
  for k-mer counting; the conceptual ancestor of our `atomicCAS` insert.
- **GenomeScope2** (<https://github.com/tbenavi1/genomescope2.0>) — profiles a
  genome (size, heterozygosity, error rate) from a k-mer **count histogram** — the
  downstream consumer of what we compute.
- **Mash** (Ondov et al., 2016) — the canonical minimiser/MinHash distance tool;
  read it for the bottom-s estimator and the Mash distance formula.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## Exercises

1. **Bigger inputs.** Run `python scripts/make_synthetic.py --genome 200000
   --reads 5000 --readlen 100` and watch the GPU kernel time vs. the CPU as the
   k-mer count grows past the launch-bound regime.
2. **k-mer histogram.** Add a histogram of counts (how many k-mers occur once,
   twice, …) and eyeball the error-vs-true peak — the input GenomeScope2 needs.
3. **Rolling encode.** Replace the O(k) per-window re-encode with an O(1) rolling
   update (shift in the new base, mask off the old) in both `kmer.h` consumers.
4. **Sort-then-reduce.** Implement the alternative counting path with
   `thrust::sort_by_key` + a run-length reduce, and check it matches the hash
   table exactly.
5. **Warp-shuffle window-min.** Rewrite `minimiser_kernel` so a warp cooperatively
   reduces 32 lanes with `__shfl_down_sync`, and confirm identical sketches.

## Limitations & honesty

- **Teaching scope.** This is a reduced-scope teaching version. We use **linear
  probing** (not the cuckoo / Robin Hood probing of production tools) and size the
  table generously (load factor < 0.5) so probe chains stay short; a real counter
  resizes, spills to disk, and handles billions of distinct keys.
- **Approximate-counting and HLL not implemented.** The catalog lists count-min
  sketch and HyperLogLog as related techniques; we implement **exact** counting and
  bottom-s **MinHash**, and describe the others in THEORY rather than coding them.
- **Synthetic data.** The committed sample is synthetic, with a planted motif and a
  hand-set A/B overlap so the output is verifiable. It is **not** a real genome and
  carries no biological or clinical meaning.
- **Timing is a teaching artifact.** On the tiny sample the GPU is launch-bound and
  may be slower than the CPU; that is expected and labelled. The GPU's advantage
  appears at the 10⁸–10⁹ k-mer scale of real read sets.
- **Not for clinical use.** Nothing here may inform diagnosis or treatment.
