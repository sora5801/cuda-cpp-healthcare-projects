# Demo — 2.11 Cryo-EM CTF Estimation (cuFFT defocus fit)

## What this demonstrates

`run_demo.ps1` (Windows) / `run_demo.sh` (Linux/CMake) will:

1. **Build** the project (links **cuFFT**) if the executable is missing.
2. **Run** it on `data/sample/micrograph_sample.txt` (a 96 × 96 synthetic
   micrograph with a known 15000 Å defocus baked in).
3. **Verify** that the GPU pipeline (cuFFT 2-D FFT → radial average → defocus
   grid-search) recovers the **same best-defocus index** as a transparent CPU
   reference (naive 2-D DFT → radial average → grid-search), and that their fit
   scores agree.
4. **Report** the recovered defocus, the fit quality (NCC), and the recovery
   error against the synthetic ground truth.

stdout (the deterministic result) is diffed against
[`expected_output.txt`](expected_output.txt); the timing lines are on stderr only.

## Canonical output

See [`expected_output.txt`](expected_output.txt). The fitter recovers
**dz = 15300 Å** for a true defocus of **15000 Å** — a 300 Å error, i.e. ~3 grid
steps of the 100 Å search, which is realistic for a single noisy micrograph and a
coarse grid (the exercises sharpen it). `RESULT: PASS` means:

- the GPU and CPU agree **exactly** on the recovered defocus *index*, and
- their normalized cross-correlation at that index agrees to `< 5e-3`.

> **Honesty note.** The full NCC *curve* (GPU vs CPU) agrees only to ~`2e-2`, not
> machine precision, because the CPU uses a **double**-precision naive DFT while
> the GPU uses **single**-precision cuFFT — so the two radial profiles genuinely
> differ at the ~1% level (PATTERNS.md §4). We verify the things that matter (the
> answer and the fit quality at the answer) and document the rest rather than
> pretend the two different-precision FFTs are bit-identical.

> The micrograph is **synthetic** (white noise shaped by a known CTF + Gaussian
> noise) — a demonstration of the CTF-estimation pattern, not a real EM analysis.
