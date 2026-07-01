# Demo — 4.26 Vessel Segmentation & Centerline Extraction

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on `data/sample/vessel_volume.txt` (a 24×16×16 volume with one
   embedded bright vessel).
3. **Verify** that the GPU Frangi-vesselness field matches the CPU reference
   (`reference_cpu.cpp`) — here **exactly** (max diff `0.000e+00`), since both run
   the same closed-form eigen + Frangi math on the same smoothed volume.
4. **Report** the peak-vesselness voxel, the segmented voxel count, a checksum,
   and the across-vessel response profile.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries timing (which varies run to run), so it is shown, not diffed.

## Canonical output

See [`expected_output.txt`](expected_output.txt). The key thing to read is the
**across-vessel profile**: a clean single-peak ridge (`… 0.16 0.63 0.63 0.16 …`)
that is zero away from the tube. That is the Frangi filter localizing the vessel:
the response is high where the local shape is tube-like (one small and two large,
negative Hessian eigenvalues) and ~0 in the flat background. `RESULT: PASS` means
the GPU and CPU vesselness fields agree.

> The volume is **synthetic** (one straight tube), and the "centerline" here is
> just the peak-response voxel — a seed, not a full graph extraction. This is a
> demonstration of the vesselness/GPU pattern, not a validated segmentation tool,
> and is **not for clinical use**.
