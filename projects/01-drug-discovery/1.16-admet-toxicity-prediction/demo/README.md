# Demo — 1.16 ADMET / Toxicity Prediction

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on `data/sample/admet_sample.txt` (24 synthetic molecules ×
   12 toxicity endpoints, 64-dim descriptors).
3. **Verify** the GPU result against the CPU reference (`reference_cpu.cpp`):
   the probability matrix must agree within `1e-9` **and** the per-endpoint
   flagged-molecule counts must match the CPU's **exactly**. Prints `PASS`/`FAIL`.
4. **Report** the per-endpoint flag counts, the single worst (most toxic)
   molecule, and time the kernels (CUDA events) vs. the CPU baseline — a
   *teaching artifact*, not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic (flag counts are integers; the
  worst-molecule pick is a deterministic argmax) and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric error (which vary run to run), so
  it is shown but never diffed.

## Canonical output

See [`expected_output.txt`](expected_output.txt) for the exact stdout the demo
asserts. A green `PASS` means the GPU probability matrix matched the CPU
reference within `1e-9` (in practice they agree to ~`5e-16`, machine precision,
because both call the *same* `__host__ __device__` math in `src/admet_core.h`)
and the GPU's integer flag counts matched the CPU's exactly.

The headline read of the result: per-endpoint flag rates **spread** from 5/24
(`hERG_block`) to 16/24 (`Ames_mutagen`), and the deliberately broadly-toxic
planted molecule **`MOL_0000`** tops the ranking (11 of 12 endpoints flagged).

> The probabilities and "toxicity" flags reflect the **synthetic** sample
> (random models, engineered to be interpretable). They carry **no chemical or
> clinical meaning** — this is study material, not a screening tool.
