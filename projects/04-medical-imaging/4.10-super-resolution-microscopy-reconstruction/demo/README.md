# Demo — 4.10 Super-Resolution Microscopy Reconstruction

## What this demonstrates

`run_demo.ps1` (Windows) / `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on `data/sample/smlm_stack.txt` (60 frames × 40×40 px, synthetic
   STORM/PALM movie).
3. **Verify** that the GPU pipeline matches the CPU reference **exactly** — same
   number of localizations, same fixed-point super-resolution image (identical
   checksum and every pixel), and mean statistics agreeing to `0` (see below).
4. **Report** the localization count, the reconstructed image dimensions, its
   fixed-point checksum, and mean fit statistics.

stdout (the deterministic digest) is diffed against
[`expected_output.txt`](expected_output.txt); the timing line is on stderr only.

## Canonical output

See [`expected_output.txt`](expected_output.txt). The pipeline detects and fits
**187 emitters** across the 60 frames and renders them into a **320×320** super-
resolution image (8× upsampled) with **123 illuminated bins**. `RESULT: PASS`
means the GPU and CPU produced the **same localizations and the same image
exactly** (`count 187/187`, `checksum` equal, `pixels_exact=yes`, `mean_err=0`).
The mean fit width (~1.35 px) is close to the true PSF sigma (1.3 px) the data was
generated with — a sanity check that the localizer is working, not just that CPU
and GPU agree.

The GPU is *slower* than the CPU here (the tiny sample is launch-bound: 60 small
kernel launches dominate); the GPU's edge appears at the 10⁴–10⁵ frames of a real
STORM run. Timing is a **teaching artifact, never a benchmark claim** (CLAUDE.md
§12; docs/PATTERNS.md §7).

> The data is a **synthetic** blinking-emitter movie, not a real microscope
> acquisition — a demonstration of GPU single-molecule localization, not a
> scientific measurement of any specimen.
