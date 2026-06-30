# Demo — 2.31 Cryo-EM Tilt-Series Alignment & Tomogram Reconstruction

## What this demonstrates

`run_demo.ps1` (Windows) / `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on `data/sample/tilt_series_sample.txt` (an analytic tilt series of
   a disc phantom, acquired over ±60° with a known per-projection drift).
3. **Align** the tilt series (recover the drift by cross-correlation), **ramp-
   filter** each projection with **cuFFT**, and **back-project** into a 2-D slice.
4. **Verify** two things: that the GPU per-pixel back-projection matches the CPU
   reference (tol `1e-3`), and that the cuFFT ramp filter matches the CPU DFT ramp
   on the interior (tol `5e-2`). Prints a single `PASS`/`FAIL`.
5. **Report** the recovered alignment shifts and reconstruction samples (center
   pixel, max, a central-row profile), and time each stage.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic (integer shifts + fixed-precision
  reconstruction samples) and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries timing and the numeric verification errors (which vary run to
  run), so it is shown but never diffed.

## Canonical output

See [`expected_output.txt`](expected_output.txt). Things to notice:

- **`estimated shifts (bins)`** recovers the injected sawtooth drift
  (`-3 -3 -2 … +3 +3`) to within ~1 bin — the tilt-series alignment working.
- The **center pixel ≈ 0.93** and the **max is near the slice center**: the
  reconstruction recovers the bright central disc.
- The **central row profile** is high across the disc and ≈ 0 (with ramp
  under/overshoot) away from it.
- **`RESULT: PASS`** means GPU == CPU back-projection within `1e-3` and the cuFFT
  ramp == CPU ramp within `5e-2` on the interior.

On this tiny sample the GPU and CPU back-projection times are comparable (the
problem is launch-bound); the GPU's edge grows with slice size and tilt count,
and a real 3-D tomogram is just a *stack* of these independent slices.

> Reconstructed values are in arbitrary phantom-density units, computed from
> **synthetic** data — a software demonstration, not a calibrated or clinical
> image. The missing wedge (only ±60° sampled) leaves visible streaking, exactly
> as in real cryo-ET.
