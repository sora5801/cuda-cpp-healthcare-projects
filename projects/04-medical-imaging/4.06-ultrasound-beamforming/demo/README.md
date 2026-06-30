# Demo — 4.6 Ultrasound Beamforming (Delay-and-Sum)

## What this demonstrates

`run_demo.ps1` (Windows) / `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on `data/sample/rf_sample.txt` — synthetic RF echoes from one
   point scatterer at a known location `(4.0, 20.0) mm`.
3. **Verify** that the GPU per-pixel delay-and-sum matches the CPU reference.
4. **Report** the recovered focal spot, the peak envelope, a lateral beam
   profile, and time GPU vs CPU.

stdout (the deterministic image summary) is diffed against
[`expected_output.txt`](expected_output.txt); the timing/error lines are on
stderr only (they vary run to run).

## Canonical output

See [`expected_output.txt`](expected_output.txt):

```
4.6 -- Ultrasound Beamforming (Delay-and-Sum)
DAS beamform: 64 elements x 384 RF samples -> 96x96 image (x by z)
focal spot (brightest pixel): (ix,iz)=(66,41)  =  (x,z)=(3.9,20.1) mm
peak envelope value = 10.5305
center pixel envelope = 0.0000
lateral profile across focal row (8 samples): 0.1896 0.0037 0.0409 0.0823 0.1705 0.0321 0.0012 0.0000
RESULT: PASS (GPU matches CPU within tol=1.0e-03)
```

The headline is the **focal spot**: the brightest pixel reconstructs to
`(3.9, 20.1) mm`, within one pixel (`dx≈0.21 mm`, `dz≈0.37 mm`) of the embedded
scatterer at `(4.0, 20.0) mm` — the beamformer correctly focused 64 independent
element echoes back onto the source. `RESULT: PASS` means the GPU image matched
the CPU reference within `1e-3`.

> On this tiny 96×96 grid the GPU kernel (~0.1 ms) already beats the serial CPU
> (~2.5 ms) several-fold; the gap widens with image size, element count, and
> frame rate. Timing is a **teaching artifact, not a benchmark claim**
> (CLAUDE.md §12). Reconstructed values are in arbitrary units — a software
> demonstration, not a clinical image.
