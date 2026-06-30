# Demo — 3.22 RNA-seq Quantification / Pseudo-alignment

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/rnaseq_ec_sample.txt` input.
3. **Verify** the GPU abundances against the CPU reference (`reference_cpu.cpp`)
   and print a clear `PASS`/`FAIL`.
4. **Time** the GPU EM loop (CUDA events) and the CPU baseline — a *teaching
   artifact*, not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the run-varying numeric error, so it is shown
  but never diffed.

## What you are looking at

The demo runs the **expectation-maximisation (EM)** transcript-quantification
algorithm (the heart of kallisto / Salmon) on a tiny synthetic problem with a
*known* answer. Each row of the table is one transcript:

- `est_counts` — expected number of reads assigned to that transcript by EM,
- `rho` — its abundance (fraction of all reads),
- `TPM` — transcripts per million (the standard length-normalised unit),
- `truth_rho` — the ground truth used to generate the data.

The `recovery: L1(...) = 0.0000` line shows the EM recovered the truth essentially
exactly, and `RESULT: PASS` confirms the GPU and CPU agree to within 1e-12 (they
agree *exactly* here — the M-step reduction uses commuting fixed-point integers).

## Expected result

```
3.22 -- RNA-seq Quantification / Pseudo-alignment
EM quantification: 6 transcripts, 9 equivalence classes, 99999 reads, 100 iterations
  id     est_counts          rho          TPM    truth_rho
  t0     30000.4242     0.300007    338083.29     0.300000
  t1      9999.5758     0.099997     75125.37     0.100000
  t2      4999.1787     0.049992     46947.64     0.050000
  t3     20000.0704     0.200003    250429.41     0.200000
  t4     14999.7509     0.149999     84518.22     0.150000
  t5     20000.0000     0.200002    204896.07     0.200000
recovery: L1(estimated rho, truth rho) = 0.0000
RESULT: PASS (GPU abundances match CPU reference)
```

The estimated `est_counts` land right on the truth read budget
(30000/10000/5000/20000/15000/20000) — notice the EM correctly split the shared
equivalence classes `{0,1}`, `{2,3}`, `{2,3,4}` using only the unambiguous
unique-region reads. The `TPM` column re-ranks by abundance-per-length: `t0`
(short, abundant) tops the TPM list even though `t5` ties it in `rho`.
