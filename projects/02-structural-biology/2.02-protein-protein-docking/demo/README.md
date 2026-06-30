# Demo — 2.2 Protein-Protein Docking (FFT rigid-body search via cuFFT)

## What this demonstrates

`run_demo.ps1` (Windows) / `run_demo.sh` (Linux/CMake) will:

1. **Build** the project (links **cuFFT**) if the executable is missing.
2. **Run** it on `data/sample/dock_sample.txt` (a synthetic receptor + a
   displaced copy of it as the ligand, on a 32³ grid).
3. **Verify** that cuFFT's `O(Ng log Ng)` correlation produces the same score
   grid as a brute-force `O(Ng²)` CPU correlation (within a documented
   round-off tolerance) **and** that the single best-scoring translation (the
   predicted dock) is identical.
4. **Report** the recovered docking translation, its shape-complementarity
   score, and whether it matches the known answer the sample was built with.

stdout (the recovered pose + score) is deterministic and diffed against
[`expected_output.txt`](expected_output.txt); the timing lines are on stderr
only (shown, not diffed).

## Canonical output

See [`expected_output.txt`](expected_output.txt):

```
2.2 -- Protein-Protein Docking
FFT rigid-body docking (Katchalski-Katzir shape correlation via cuFFT)
grid: 32x32x32 voxels @ 1.50 A/voxel   receptor atoms: 1954   ligand atoms: 1954
best translation (voxels): t = (-3, -2, 1)
best shape-complementarity score: 77294.0000
known-answer translation:  t = (-3, -2, 1)  -> RECOVERED
RESULT: PASS (cuFFT score grid matches CPU within 5e-01; best pose identical)
```

The ligand was displaced by `D = (3, 2, −1)` voxels, so the search correctly
recovers the translation `t = −D = (−3, −2, 1)` that slides it back onto the
receptor. `RESULT: PASS` means cuFFT's score grid matched the brute-force CPU
grid and the argmax pose is identical. On the RTX 2080 SUPER the cuFFT route
runs in ~0.5 ms versus ~1.3 s for the brute-force CPU correlation — the
`O(Ng²) → O(Ng log Ng)` win that makes FFT docking practical (a *teaching
artifact*, not a benchmark claim; see THEORY §7).

> The input is **synthetic** (a geometric test object with a known answer), a
> demonstration of the FFT rigid-body docking pattern — not a real complex
> prediction. See `data/README.md` and `THEORY.md` for the honest scope.
