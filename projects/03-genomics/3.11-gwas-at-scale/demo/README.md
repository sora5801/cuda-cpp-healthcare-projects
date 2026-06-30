# Demo — 3.11 GWAS at Scale

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/gwas_sample.txt` (a synthetic cohort).
3. **Verify** the GPU result against the CPU reference (`reference_cpu.cpp`):
   - the **GRM** (`(1/M)·Z·Zᵀ`, built with **cuBLAS DGEMM**) matches entrywise,
   - the **per-SNP χ²** (the association scan) matches per SNP,
   and prints a clear `PASS`/`FAIL`.
4. **Recover the planted answer:** the synthetic data injects 5 *causal* SNPs;
   the demo confirms all 5 are ranked in the **top 10** by association strength.
5. **Time** the standardize kernel, the cuBLAS DGEMM, the scan, and the CPU
   baseline — a *teaching artifact*, not a benchmark claim.

The program splits its output deliberately (PATTERNS.md §3):

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries timing and the numeric verification error (which vary run to
  run), so it is shown but never diffed.

## What to look for

- `recovered in top 10: 5` — every injected causal SNP was found.
- The top 5 rows are all marked `CAUSAL`, in descending `χ²` / `−log10p` order,
  with `beta` magnitudes tracking the injected effect sizes.
- `RESULT: PASS` — the GPU and CPU agree within tolerance.
- On **stderr**, `GRM worst entry diff ≈ 6.7e-16` and `assoc worst chi2 diff ≈
  2e-13`: the two implementations agree to near machine precision even though
  cuBLAS sums the matrix multiply in a different order (THEORY §5).

## Expected result (stdout)

```
3.11 -- GWAS at Scale
cohort: N=200 individuals, M=60 SNPs (synthetic)
GRM: mean diagonal=1.0099, max |off-diagonal|=0.6009 at (107,1)
causal SNPs injected: 5 ; recovered in top 10: 5
top 10 associated SNPs (rank: id chi2 -log10p beta causal):
   1: rs1038   chi2= 137.1956  -log10p=30.9615  beta=+3.92788  CAUSAL
   2: rs1034   chi2=  86.8905  -log10p=19.9404  beta=-3.36822  CAUSAL
   3: rs1025   chi2=  62.0824  -log10p=14.4823  beta=+2.79403  CAUSAL
   4: rs1022   chi2=  19.4789  -log10p= 4.9926  beta=-1.71807  CAUSAL
   5: rs1009   chi2=  10.1061  -log10p= 2.8304  beta=+1.21522  CAUSAL
   6: rs1041   chi2=   4.3984  -log10p= 1.4440  beta=-0.79529  -
   7: rs1018   chi2=   3.4252  -log10p= 1.1924  beta=+0.74028  -
   8: rs1017   chi2=   2.8672  -log10p= 1.0438  beta=+0.66102  -
   9: rs1015   chi2=   2.3434  -log10p= 0.9003  beta=-0.62233  -
  10: rs1019   chi2=   2.2214  -log10p= 0.8661  beta=-0.57557  -
RESULT: PASS (GPU GRM and association match CPU within tol)
```

(Timings on stderr will differ on your machine — that is expected and not diffed.)
