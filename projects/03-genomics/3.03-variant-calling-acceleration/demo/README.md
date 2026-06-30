# Demo — 3.3 Variant Calling Acceleration

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/reads_haplotypes_sample.txt`.
3. **Verify** the GPU PairHMM log-likelihood matrix against the CPU reference
   (`reference_cpu.cpp`) and print a clear `PASS`/`FAIL`.
4. **Time** the kernel (CUDA events) and the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

The computation is the **PairHMM forward algorithm**, the dominant cost in
germline variant calling: for every (read, haplotype) pair it computes
`log10 P(read | haplotype)`, then assigns each read to its most-likely
haplotype. The synthetic sample draws all 8 reads from haplotype 0 (the
"truth"), so a correct caller assigns every read to haplotype 0.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt). It uses `double` math shared by
  the CPU and GPU (`src/pairhmm_core.h`), so Release and Debug produce identical
  bytes.
- **stderr** carries the timing and the numeric error (which vary run to run), so
  it is shown but never diffed.

## Expected result

```
3.3 -- Variant Calling Acceleration
PairHMM forward: 8 reads x 3 haplotypes (read_len=20, hap_len=30)
per-read best haplotype (argmax log10 P(read|hap)):
  read  0 -> hap 0   log10L = -8.499181
  read  1 -> hap 0   log10L = -1.554920
  read  2 -> hap 0   log10L = -1.555041
  read  3 -> hap 0   log10L = -1.555627
  read  4 -> hap 0   log10L = -1.555627
  read  5 -> hap 0   log10L = -1.555627
  read  6 -> hap 0   log10L = -4.882619
  read  7 -> hap 0   log10L = -5.009819
reads assigned to truth haplotype 0: 8 of 8
RESULT: PASS (GPU matches CPU within tol=1.0e-09)
```

### How to read it

- Each line gives a read's best haplotype and its log10 likelihood. All reads
  pick **hap 0**, the truth — that is the headline success criterion.
- Reads with several simulated errors (e.g. read 0) have a much lower likelihood
  (`-8.5` vs `-1.55`) but are still correctly assigned: the alternative
  haplotypes differ by a SNP, which costs even more likelihood than a sequencing
  error does.
- `RESULT: PASS` means the GPU's log-likelihood matrix matched the CPU
  reference to within `1.0e-9` (the actual `max_abs_err` printed on stderr is
  ~`1.8e-15`, essentially machine precision, because both paths run the same
  IEEE-754 `double` operations).

A typical **stderr** (varies per machine; not diffed):

```
[data]   source: ...reads_haplotypes_sample.txt  (8 reads, 3 haplotypes)
[model]  delta(gap-open)=0.0015  epsilon(gap-extend)=0.1
[timing] CPU reference: 0.3 ms   GPU kernel: 10.0 ms
[verify] max_abs_err = 1.776e-15  (tolerance 1.0e-09)
```

On this tiny input the GPU is *slower* than the CPU — the kernel time is
dominated by launch overhead, not arithmetic. That is expected and honest: the
GPU's advantage appears when there are thousands of read-haplotype pairs, as in
real variant calling. See `../THEORY.md` "honest timing".
