# Demo — 2.32 Protein Folding Pathway Extraction (Transition Path Sampling)

## What this demonstrates

`run_demo.ps1` (Windows) / `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on `data/sample/tps_params.txt` — 4096 independent TPS shooting
   moves on a 5 kT double-well folding landscape (basins at x = 0.1 "unfolded"
   and x = 0.9 "folded", barrier at x = 0.5).
3. **Verify** that the GPU and CPU produce the **exact same** integer tallies
   (transition-path count, committor histogram) — because both run the identical
   shooting moves (shared RNG + Brownian dynamics) and integer atomic adds
   commute.
4. **Report** the fraction of reactive transition paths, the committor curve
   p_B(x), and the recovered transition-state bin; plus timing.

stdout (the deterministic stats + committor curve) is diffed against
[`expected_output.txt`](expected_output.txt); the timing line is on stderr only.

## Canonical output

See [`expected_output.txt`](expected_output.txt). What to look for:

- **`RESULT: PASS`** means the GPU tally equals the CPU tally **exactly** (`0
  mismatches`) — the correctness guarantee.
- **The committor curve rises monotonically** from `0` (% reaching the folded
  basin) near the unfolded basin to `100` near the folded basin. That S-shaped
  rise is the signature of a committor along a good reaction coordinate.
- **The transition-state bin is `10`**, which sits at x ≈ 0.5 = the barrier top —
  exactly where the committor p_B crosses 1/2. Recovering the barrier-top as the
  p_B = 0.5 isosurface is the scientific point of committor analysis, and it is a
  *known-answer* check beyond mere CPU==GPU agreement (PATTERNS.md §4).
- About **13.5 %** of shots are accepted as transition paths (they connect the
  two basins) — the rest fall back into the basin they came from.

> This is a **simplified, synthetic** teaching model: a **1-D** reaction
> coordinate on an analytic double-well surface with **overdamped Langevin
> dynamics** and a simplified shooting move — **not** an all-atom MD engine and
> not a source of real folding kinetics.
