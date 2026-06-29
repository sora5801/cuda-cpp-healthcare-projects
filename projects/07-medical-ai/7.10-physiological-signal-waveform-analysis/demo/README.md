# Demo — 7.10 Physiological Signal & Waveform Analysis

## What this demonstrates

`run_demo.ps1` (Windows) / `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on `data/sample/ecg_sample.txt` (a noisy synthetic ECG, 2048 samples).
3. **Verify** the GPU tiled 1-D convolution matches the CPU reference.
4. **Report** the filtered peak, how much was removed (RMS of `x − filtered`),
   and a few filtered samples.

stdout (filtered samples) is deterministic and diffed against
[`expected_output.txt`](expected_output.txt); the timing line is on stderr only.

## Canonical output

See [`expected_output.txt`](expected_output.txt). The Gaussian low-pass smooths
the high-frequency noise (the reported RMS removed is non-zero), and
`RESULT: PASS` means the GPU and CPU filtered signals agree to ~`1e-7`.

> The signal is **synthetic** and the filter is a generic Gaussian low-pass — a
> demonstration of the 1-D convolution / shared-memory-tiling pattern, not a
> validated ECG analysis. (A real ECG pipeline uses a band-pass that preserves
> the QRS complex — see THEORY.)
