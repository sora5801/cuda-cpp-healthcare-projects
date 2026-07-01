# Demo — 4.21 MR Fingerprinting Reconstruction

## What this demonstrates

One command builds (if needed) and runs the MR Fingerprinting reconstruction on
the committed **synthetic** sample, then checks the GPU result against the CPU
reference and prints a labeled, deterministic summary.

The demo shows the whole MRF pipeline:

1. **Build the dictionary** — simulate every `(T1, T2)` atom's signal fingerprint
   (one GPU thread per atom) and L2-normalize it.
2. **Normalize the voxel signals** — one thread per voxel.
3. **Match** — form the entire `V × D` cosine-score matrix with **one cuBLAS
   SGEMM**, then pick each voxel's best-scoring atom with a per-voxel `argmax`
   kernel. The matched atom's `(T1, T2)` are the reconstructed tissue parameters.
4. **Verify** — the GPU dictionary matches the CPU's, and the per-voxel best-atom
   **index** matches the CPU **exactly** (the strong correctness check).

Because each synthetic voxel was drawn from a **known** atom, the headline result
is **reconstruction accuracy**: how many of the 64 voxels recover their true atom
(and the median `T1`/`T2` error).

## Run it

```powershell
# Windows (PowerShell), from the project folder:
./demo/run_demo.ps1
```

```bash
# Linux/macOS (uses the optional CMake build):
./demo/run_demo.sh
```

## Expected output

Deterministic result → **stdout** (diffed against
[`expected_output.txt`](expected_output.txt)); timing + verification detail →
**stderr** (shown, not diffed, because timings vary run to run). Success looks
like:

```
4.21 -- MR Fingerprinting Reconstruction
problem: T=120 frames, D=22 dictionary atoms, V=64 voxels (synthetic)
match accuracy: 64/64 voxels matched their ground-truth atom
median |T1 error| = 0.000 ms ; median |T2 error| = 0.000 ms
recovered T1 map range: [250.0, 2800.0] ms ; T2 map range: [25.0, 160.0] ms
first 5 voxels (voxel: truth_atom -> matched_atom  T1  T2  cos  PD):
  v00: truth=  3 -> atom=  3  T1= 250.0  T2= 100.0  cos=0.9996  PD=1.7314
  ...
RESULT: PASS (GPU dictionary + argmax match CPU; indices exact)
```

The **stderr** stream additionally reports the per-stage timings (dictionary
build, signal normalize, cuBLAS SGEMM, argmax), the worst dictionary/cosine
difference vs. the CPU, the number of best-atom index mismatches (0), and a
spot-check of the SGEMM's voxel-0 score row against a hand-rolled CPU inner
product.

## Notes

- **Timing is a teaching artifact, not a benchmark.** On this tiny sample the GPU
  is launch/copy-bound; the SGEMM's advantage grows as `O(V·D·T)` — with real MRF
  sizes (`V ~ 10⁵`, `D ~ 10⁵`, `T ~ 10³`) the match is ~10¹¹ inner products, which
  is exactly where a single big GEMM crushes a serial CPU loop.
- All data is **synthetic** and labeled as such; no clinical claims.
