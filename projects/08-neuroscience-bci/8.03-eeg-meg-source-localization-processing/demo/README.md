# Demo — 8.03 EEG/MEG Spectral Processing (cuFFT)

## What this demonstrates

`run_demo.ps1` (Windows) / `run_demo.sh` (Linux/CMake) will:

1. **Build** the project (links **cuFFT**) if the executable is missing.
2. **Run** it on `data/sample/eeg_sample.txt` (8 channels × 256 samples).
3. **Verify** that cuFFT's batched FFT gives the same **band powers** as a naive
   CPU DFT.
4. **Report** the delta/theta/alpha/beta/gamma power of each channel and its
   dominant band.

stdout (band powers) is deterministic and diffed against
[`expected_output.txt`](expected_output.txt); the timing line is on stderr only.

## Canonical output

See [`expected_output.txt`](expected_output.txt). Each synthetic channel was built
with a known dominant rhythm, and the demo recovers it (ch0→alpha, ch1→beta,
ch2→theta, ch3→delta, ch4→gamma, ...). `RESULT: PASS` means cuFFT's band powers
match the naive DFT to ~`1e-6` relative. cuFFT (O(N log N), batched) is ~16× faster
than the O(N²) DFT here, and the gap explodes with N and channel count.

> The signal is **synthetic** (clean sinusoids + noise) — a demonstration of the
> cuFFT spectral-processing pattern, not a real EEG analysis.
