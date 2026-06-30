# Demo — 2.20 Heterogeneous Cryo-EM Reconstruction (3D Variability)

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/volumes.txt` (24 synthetic particle
   volumes hiding one continuous "z-slide" conformational motion).
3. **Verify** the GPU 3DVA result (mean volume, Gram eigenvalues via cuSOLVER,
   principal component PC1, per-particle latent coordinates) against the CPU
   reference (`reference_cpu.cpp`, which uses a transparent Jacobi eigensolver)
   and print a clear `PASS`/`FAIL`.
4. **Time** each GPU stage (CUDA events) and the CPU baseline — a *teaching
   artifact*, not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric verification errors (which vary
  run to run), so it is shown but never diffed.

## Expected result

```
2.20 -- Heterogeneous Cryo-EM Reconstruction (3D Variability)
3DVA (PCA on volumes): N=24 particles, G=6, D=216 voxels/volume
top-3 mode variances (eigenvalues, descending): 1.35169 0.14476 0.00472
PC1 variance explained: 0.8994
latent z along PC1 (8 sampled particles): +1.7639 +1.4561 +1.0194 +0.4876 -0.2887 -0.8507 -1.3289 -1.7556
PC1 vs ground-truth conformation |corr|: 0.9974
RESULT: PASS (GPU 3DVA matches CPU reference within tol)
```

## How to read it

- **top-3 mode variances** are the largest eigenvalues of the volume covariance.
  They drop off a cliff (1.35 ≫ 0.14 ≫ 0.005): the data really has **one**
  dominant axis of variation, exactly as we synthesized it.
- **PC1 variance explained = 0.8994** — ~90% of all volume variation lives along
  a single principal component. That is the 3DVA headline: the molecule's
  flexibility is (almost) one-dimensional here.
- **latent z** is each particle's coordinate along PC1. The 8 sampled values
  sweep monotonically from +1.76 to −1.76 — i.e. PC1 *is* the blob-slide axis,
  and the particles spread out evenly along it.
- **|corr| = 0.9974** — the recovered latent coordinate tracks the hidden
  ground-truth conformational parameter almost perfectly. 3DVA found the motion.
  (Sign is arbitrary for a principal component, so we report `|corr|`.)

The `[verify]` lines on stderr show GPU and CPU agree to ~1e-15 (machine
precision), because both paths run the *same* shared `__host__ __device__` math
and only the eigensolver differs (cuSOLVER `Dsyevd` vs Jacobi), tolerance `1e-9`.
