# Demo — 4.01 CT Reconstruction (Filtered Backprojection)

## What this demonstrates

`run_demo.ps1` (Windows) / `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on `data/sample/sinogram_sample.txt` (an analytic sinogram of a
   disc phantom).
3. **Verify** that the GPU per-pixel backprojection matches the CPU reference.
4. **Report** reconstructed image samples (center pixel, max, a central-row
   profile) and time GPU vs CPU.

stdout (image samples) is deterministic and diffed against
[`expected_output.txt`](expected_output.txt); the timing line is on stderr only.

## Canonical output

See [`expected_output.txt`](expected_output.txt). The center pixel reconstructs
to ≈ 1.0 (the main disc's density), the profile is flat across the disc and ≈ 0
outside it, and `RESULT: PASS` means the GPU image matched the CPU image within
`1e-3`. On the sample the GPU backprojection is ~30× faster than the CPU — a
genuine win that grows with image size and projection count.

> Reconstructed values are in arbitrary phantom-density units — a software
> demonstration, not a calibrated clinical image.
