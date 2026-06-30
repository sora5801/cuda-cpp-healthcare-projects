# Demo — 2.3 Cryo-EM Single-Particle Reconstruction

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed synthetic sample (`data/sample/cryoem_sample.txt`).
3. **Verify** the GPU result against the CPU reference (`reference_cpu.cpp`):
   - every per-particle orientation assignment matches **exactly** (integer), and
   - the reconstructed density matches the CPU density within `1e-4`
     (in practice `0.0` — the shared `__host__ __device__` math is bit-identical).
4. **Time** the two GPU kernels (CUDA events) and the CPU baseline — a *teaching
   artifact*, not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries timing and the numeric error (which vary run to run), so it
  is shown but never diffed.

## What you are looking at

- **E-step (projection matching).** Each of the 120 noisy particles is scored
  against all 60 reference projections; the best-correlating angle is its assigned
  orientation. This is the `O(N·M)` sweep the GPU parallelizes (one thread per
  particle). The reported **70.8% exact-angle accuracy** is honest for σ≈15% noise
  at 3° angular sampling — **93% of particles land within ±1 angle (±3°)** of
  truth; the few "misses" are almost all nearest-neighbour, not random (see THEORY).
- **M-step (back-projection).** The assigned 1-D profiles are smeared back into a
  64×64 density (one thread per output pixel, a gather). The **reconstruction-vs-
  truth NCC = 0.8764** shows the original phantom is genuinely recovered, blur and
  all.
- **density digest.** Five fixed interior samples of the reconstruction — a stable
  fingerprint that pins the exact numbers so the demo catches any regression.

## Expected result

```
2.3 -- Cryo-EM Single-Particle Reconstruction
2D single-particle reconstruction (synthetic phantom)
geometry: image 64x64, 60 reference angles, 120 particles
E-step (projection matching, O(N*M)=7200 comparisons):
  orientation recovery accuracy = 70.8% (85/120 exact-angle hits)
M-step (back-projection into 64x64 density):
  reconstruction-vs-truth NCC = 0.8764
  density digest: centre=12.7030  q1=7.9486  q2=3.4080  q3=4.2988  q4=5.5459
RESULT: PASS (GPU matches CPU: 120/120 assignments exact, density within tol=1.0e-04)
```

The `stderr` timing line (kernel ms, CPU ms) will differ on your machine — that is
expected and is exactly why it is kept out of the diffed stdout.
