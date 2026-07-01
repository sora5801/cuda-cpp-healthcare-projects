# Demo — 4.13 Photoacoustic Image Reconstruction

## What this demonstrates

`run_demo.ps1` (Windows) / `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on `data/sample/pa_sample.txt` — a synthetic ring-array photoacoustic
   acquisition with three planted point absorbers.
3. **Verify** that the GPU per-pixel delay-and-sum backprojection matches the CPU
   reference (`src/reference_cpu.cpp`) within tolerance.
4. **Report** the reconstructed peak location, a center-row profile, and time GPU
   vs CPU.

stdout (image samples) is deterministic and diffed against
[`expected_output.txt`](expected_output.txt); the timing/error line is on stderr
only (it varies run to run).

## Canonical output

See [`expected_output.txt`](expected_output.txt):

```
4.13 -- Photoacoustic Image Reconstruction
2-D delay-and-sum backprojection
64 sensors x 512 samples -> 96x96 image (c=1500.0 m/s, dt=5.000e-08 s)
peak value = 33.2748 at pixel (px,py)=(47,47) = (x,y)=(-0.0001,-0.0001) m
center-row profile (8 samples): 0.1328 2.0825 1.8289 3.5928 4.2744 1.7030 1.8482 0.1791
RESULT: PASS (GPU matches CPU within tol=1.0e-03)
```

**How to read it.** The brightest reconstructed pixel is `(47,47)`, i.e. world
`(≈0, ≈0)` metres — exactly where the **strongest** planted absorber sits, so the
reconstruction recovered the ground truth. The center-row profile rises toward the
middle (crossing near the sources) and falls off toward the edges: the classic
delay-and-sum point response. `RESULT: PASS` means the GPU image matched the CPU
image within `1e-3` (they differ only at `~3e-4` from GPU fused-multiply-add; see
THEORY.md §5). On this sample the GPU reconstruction is several times faster than
the CPU — a genuine win that grows with image size and sensor count.

> Reconstructed values are in arbitrary pressure units from **synthetic** data — a
> software demonstration, not a calibrated or clinical image.
