# Demo — 3.4 Nanopore Basecalling (CTC greedy decode)

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/reads_sample.txt` (4 synthetic reads).
3. **Decode** each read's posterior matrix into DNA bases via greedy CTC, on both
   the GPU (`src/kernels.cu`) and the CPU reference (`src/reference_cpu.cpp`).
4. **Verify** they agree **exactly** — same length, base string, and checksum for
   every read — and print a clear `PASS`/`FAIL`.
5. **Time** the kernel (CUDA events) and the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing, the data path, and notes (which vary run to run or
  by machine), so it is shown but never diffed.

## What to look for

Each line `read r: T=… len=… checksum=… seq=…` shows a read's decoded sequence. The
synthetic sample **plants known sequences**, so you should see them recovered exactly:

- `read 0 … seq=ACGTACGT` — all distinct neighbours.
- `read 1 … seq=AACCGGTT` and `read 3 … seq=TTTTGGGG` — **homopolymers**, the case
  CTC's blank symbol exists to handle. The decoder keeps the doubled/repeated bases.

The final `CPU/GPU agreement: 4/4 reads identical` and `RESULT: PASS` confirm the GPU
decode matches the trusted CPU decode bit-for-bit (tolerance `0`, because both run the
identical integer decode in `src/ctc_core.h`).

## Expected result (stdout)

```
3.4 -- Nanopore Basecalling (CTC greedy decode)
decoded 4 reads from posterior matrices (C=5 classes: -ACGT)
  read 0: T=32  len=8  checksum=6c2d4a63  seq=ACGTACGT
  read 1: T=32  len=8  checksum=faa0aca5  seq=AACCGGTT
  read 2: T=28  len=7  checksum=6fbc9be0  seq=GATTACA
  read 3: T=32  len=8  checksum=bb5f5785  seq=TTTTGGGG
total called bases (all reads): 31
CPU/GPU agreement: 4/4 reads identical
RESULT: PASS (GPU matches CPU exactly; tol = 0, integer decode)
```

The stderr lines (timing, data source, the "network is out of scope" note) are shown
by the demo but are not part of the diff, since timings vary run to run.
