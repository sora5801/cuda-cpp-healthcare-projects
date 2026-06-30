# Demo — 3.17 CRISPR Guide Design & Off-Target Scoring

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on `data/sample/guide_genome_sample.txt` (1 guide vs a 418-base
   synthetic genome → 396 candidate windows).
3. **Verify** the GPU per-window scores against the CPU reference
   (`reference_cpu.cpp`) and print a clear `PASS`/`FAIL`.
4. **Report** the recovered on-target site and the **top-5 off-targets ranked by
   CFD score**, plus the guide's aggregate specificity.
5. **Time** the kernel (CUDA events) and the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt). The host does the ranking and the
  CFD summation in a **fixed order**, so the summary is reproducible.
- **stderr** carries the timing and the numeric error (which vary run to run), so
  it is shown but never diffed.

## Expected result

See [`expected_output.txt`](expected_output.txt) for the exact stdout the demo
asserts. The scan recovers the single on-target site at position 40 and ranks the
off-targets by CFD — note how the **1-mismatch distal** site scores ~0.86 while
the **1-mismatch seed** site scores only 0.05: that is the seed effect, the most
important fact in CRISPR off-target biology.

A green `PASS` means the GPU result matched the CPU reference: the integer
mismatch counts are **identical** and the CFD scores agree to within `1e-12`
(they are computed by the same `__host__ __device__` function — see `THEORY.md`
§"How we verify correctness").

> The genome and the CFD weights are **synthetic** (a teaching model, not the
> published Doench-2016 table); the scores carry no biological meaning.
