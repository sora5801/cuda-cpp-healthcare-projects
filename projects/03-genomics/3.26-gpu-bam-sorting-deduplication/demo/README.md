# Demo — 3.26 GPU BAM Sorting & Deduplication

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on `data/sample/reads_sample.txt` (2,000 synthetic aligned reads).
3. **Coordinate-sort** the reads on the GPU (`thrust::sort_by_key`, a radix sort
   on a packed `(ref, pos, strand)` key) and **mark duplicates** on the GPU
   (`thrust::reduce_by_key`, a segmented "keep the best copy per fragment").
4. **Verify** both results against the CPU reference (`reference_cpu.cpp`) — the
   sort order and the duplicate flags must match **exactly** (all integers, total
   orders, so no tolerance is needed) — and print `PASS`/`FAIL`.
5. **Time** the GPU sort and dedup (CUDA events) and the CPU baseline — a
   *teaching artifact*, not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing (which varies run to run), so it is shown but
  never diffed.

## Expected result

See [`expected_output.txt`](expected_output.txt). The demo sorts all 2,000 reads
into genome order, reports a stable FNV-1a digest of that order (a compact proof
the GPU order equals the CPU order), shows the first 8 sorted reads — note the
four reads sharing `ref=0, pos=46030, strand=1, mate=8338`, a duplicate cluster —
and flags exactly **358 duplicates** (the number planted by the synthetic
generator), keeping 1,642 reads. `RESULT: PASS` means the GPU sort **and** dedup
matched the CPU reference exactly (`0 sort mismatches`, `0 dup-flag mismatches`).

On this tiny sample the GPU is launch/copy bound, so it is not faster than the
CPU here; the radix-sort advantage grows with read count (real BAMs hold
10⁸–10⁹ reads). The timing line says so.

> The data is **synthetic** aligned reads with a deliberately planted duplicate
> structure — a demonstration of GPU sort + dedup, not a clinical analysis.
