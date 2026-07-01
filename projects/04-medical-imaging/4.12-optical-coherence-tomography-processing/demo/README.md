# Demo — 4.12 Optical Coherence Tomography Processing (SD-OCT)

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project (links **cuFFT**) if the executable is missing.
2. **Run** it on `data/sample/oct_bscan.txt` — a synthetic B-scan of 32 A-scans ×
   256 spectral samples with **injected dispersion** (`a2=18, a3=9`).
3. **Reconstruct** the B-scan on the GPU (custom preprocessing/dispersion kernel →
   **batched cuFFT** → magnitude kernel) and on the CPU (naive DFT reference).
4. **Verify** two ways: the per-A-scan **peak-depth index** matches the CPU
   *exactly* (integer argmax), and the normalised images agree within `atol=2e-4`.
5. **Report** the recovered peak depths and a small ASCII rendering of the
   reconstructed cross-section.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric image error (which vary run to
  run), so it is shown but never diffed.

## Canonical output

See [`expected_output.txt`](expected_output.txt). The synthetic B-scan has a gently
curved bright "surface" reflector; the demo recovers that arc as the per-A-scan
peak depth (8 → 14 → 8, symmetric across the field) and the ASCII image shows the
curved layer. `RESULT: PASS` means cuFFT's batched reconstruction reproduces the
naive-DFT reference (peak depths exact, images to ~`1e-7` here). The GPU path
(custom kernels + one batched cuFFT call) is several × faster than the O(N²) DFT,
and the gap explodes with N and A-scan count — real B-scans are 2048 × 2048.

> The signal is **synthetic** (reflectors at known depths + injected dispersion +
> noise) — a demonstration of the SD-OCT reconstruction + dispersion-compensation
> pattern, not a real clinical OCT acquisition, and of no diagnostic meaning.
