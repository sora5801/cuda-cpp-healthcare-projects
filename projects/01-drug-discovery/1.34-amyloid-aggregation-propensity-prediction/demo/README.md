# Demo — 1.34 Amyloid / Aggregation Propensity Prediction

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/amyloid_sample.fasta` (4 short
   **synthetic** proteins).
3. **Scan** every sequence: map each residue to its intrinsic β-aggregation
   propensity, smooth with a 7-residue sliding-window mean on the GPU
   (shared-memory tiled, one block per protein), threshold to find
   aggregation-prone regions (APRs), and rank the proteins.
4. **Verify** the GPU result against the CPU reference (`reference_cpu.cpp`):
   the smoothed profiles and `peak_score` agree to ≤ `1e-5`, and the integer
   fields (`peak_pos`, `prone_count`, `longest_apr`) agree **exactly**. Prints a
   clear `PASS`/`FAIL`.
5. **Time** the kernel (CUDA events) and the CPU baseline — a *teaching
   artifact*, not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric error (which vary run to run), so
  it is shown but never diffed.

## Expected result

```
1.34 -- Amyloid / Aggregation Propensity Prediction
sequence-based APR scan: 4 proteins, window W=7, threshold=0.55
(synthetic sequences; intrinsic propensity scale -- not for clinical use)
rank  protein                         len  peak  pos  resi  prone  APR
   1  SYNTH_AGG_polyV [synthetic; br   36 0.993    8   I     26   26
   2  SYNTH_AGG_core  [synthetic; bu   60 0.979   23   L     20   20
   3  SYNTH_MIXED [synthetic; near-t   40 0.640    1   K     18    1
   4  SYNTH_SOLUBLE_charged [synthet   50 0.133   47   E      0    0
top hit 'SYNTH_AGG_polyV [synthetic; broad aliphatic stretch]' smoothed profile (12 pts): 0.150 0.386 0.736 0.993 0.979 0.979 0.986 0.993 0.986 0.857 0.500 0.138
RESULT: PASS (GPU matches CPU: max_abs_err <= 1e-05, integer fields exact)
```

## How to read it

- **rank** — proteins sorted by peak smoothed propensity (most aggregation-prone
  first). The two designed hydrophobic constructs top the list; the charged
  negative control (`SYNTH_SOLUBLE_charged`) sits at the bottom with **0** prone
  residues, exactly as designed.
- **peak / pos / resi** — the highest smoothed score, the residue index where it
  occurs, and that residue's one-letter code.
- **prone / APR** — how many residues exceed the threshold, and the longest
  *contiguous* aggregation-prone region. Note `SYNTH_MIXED` has 18 prone
  residues but an APR length of only **1**: its prone residues alternate with
  soluble ones, so it never forms a long, fibril-nucleating stretch — a real
  teaching point about why *contiguity* matters, not just the count.
- **smoothed profile** — 12 evenly-spaced points of the top hit's profile; you
  can see it rise well above the `0.55` threshold across the hydrophobic core.

Everything here is **synthetic** and **not for clinical use** — see
`data/README.md` and `THEORY.md`.
