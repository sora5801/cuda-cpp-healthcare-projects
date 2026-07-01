# Demo — 4.19 Motion-Compensated 4D-CT Reconstruction

## What this demonstrates

`run_demo.ps1` (Windows) / `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on `data/sample/sinogram4d_sample.txt` — an analytic sinogram of a
   *breathing* disc phantom (a nodule that moves with the respiratory phase).
3. **Reconstruct** the image two ways, on both CPU and GPU:
   - **naive 4D-FBP** — ignores motion → the moving nodule smears (motion blur);
   - **motion-compensated** — warps each phase by its Deformation Vector Field
     (DVF) → the nodule re-focuses.
4. **Verify** each GPU image matches its CPU reference within `1e-3` (in practice
   the agreement is ~`2e-6`, near single-precision machine epsilon).
5. **Report** the headline result — **peak recovery** of the moving nodule — plus
   a column profile through it, and time GPU vs CPU.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric error (which vary run to run), so
  it is shown but never diffed.

## Expected result

See [`expected_output.txt`](expected_output.txt). The key lines:

```
naive 4D-FBP  peak = 0.8966 ...      <- motion smeared the nodule below its true density
motion-comp   peak = 1.0413 ...      <- motion compensation recovered it toward 1.0
peak recovery (MCR / naive) = 1.1615x   (true nodule density = 1.0)
RESULT: PASS (GPU matches CPU within tol=1.0e-03; MCR recovers the moving nodule)
```

The naive reconstruction's peak (≈ 0.90) sits **below** the nodule's true density
(1.0) because breathing spread its energy across every phase's position. Motion
compensation re-aligns the phases and the peak climbs back to ≈ 1.04 — recovering
the true intensity within a few percent, and moving the peak to the reference-frame
location. That "peak recovers toward the known density" is a physical check on the
science, not just CPU==GPU agreement.

On this tiny problem the GPU reconstruction is already several times faster than the
CPU; the gap widens with image size and projection count (see stderr timing).

> Reconstructed values are in arbitrary phantom-density units — a software
> demonstration of the motion-compensation idea, **not** a calibrated clinical image.
