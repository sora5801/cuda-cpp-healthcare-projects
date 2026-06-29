# Demo — 5.01 Monte Carlo Dose Calculation (simplified slab)

## What this demonstrates

`run_demo.ps1` (Windows) / `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it with `data/sample/mc_params.txt` (simulate 262,144 photon histories
   through a 20 cm slab).
3. **Verify** that the GPU and CPU produce the **exact same** integer depth-dose
   tally — because both run the identical histories (shared RNG) and integer
   atomic adds commute.
4. **Report** the depth-dose histogram, the deposited fraction, and timing.

stdout (the deterministic histogram) is diffed against
[`expected_output.txt`](expected_output.txt); the timing line is on stderr only.

## Canonical output

See [`expected_output.txt`](expected_output.txt). The depth-dose falls off with
depth (more interactions near the entrance, attenuation reduces deeper bins), and
`RESULT: PASS` means the GPU tally equals the CPU tally **exactly** (`0
mismatches`). On the sample the GPU is ~10× faster than the CPU, a gap that grows
with the history count (clinical plans run 10⁹–10¹⁰ histories).

> This is a **simplified, synthetic** teaching model (1-D, integer quanta, no real
> cross sections or electron transport) — not a dose engine.
