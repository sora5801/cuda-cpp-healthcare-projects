# Demo — 2.21 Protein-Nucleic Acid Docking & Co-Folding

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/complex_sample.txt` — a tiny synthetic
   protein-nucleic-acid complex with a *planted native pose*.
3. **Search** the full 6-D pose grid (24 orientations × 3×3×3 translations = 648
   poses) on the GPU, one thread per pose, and on the CPU reference.
4. **Verify** the GPU result against the CPU reference — **exactly**, because every
   pose score is integer arithmetic (the demo prints `648/648 poses agree`).
5. **Rank** the poses and print the top 5; the #1 hit is the native pose.
6. **Time** the kernel (CUDA events) and the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing (which varies run to run), so it is shown but never
  diffed.

## Expected result

```
2.21 -- Protein-Nucleic Acid Docking & Co-Folding
rigid-body docking: protein (25 atoms) vs nucleic-acid ligand (9 atoms)
pose space: 24 orientations x 3x3x3 translations = 648 poses
top-5 poses by interface score:
  #1  pose 312  score = 340  rot = 0  t = (0.000, 0.000, 3.500) A
  #2  pose 316  score = 240  rot = 4  t = (0.000, 0.000, 3.500) A
  #3  pose 240  score = 190  rot = 0  t = (0.000, -3.500, 3.500) A
  #4  pose 244  score = 190  rot = 4  t = (0.000, -3.500, 3.500) A
  #5  pose 302  score = 190  rot = 14  t = (-3.500, 0.000, 3.500) A
RESULT: PASS (GPU matches CPU exactly: 648/648 poses agree)
```

**How to read it.** The winner — **pose 312, rotation 0 (identity), translation
(0, 0, 3.5 Å)** — is exactly the native pose planted in `make_synthetic.py`: the
ligand seated one contact shell above the protein's charged patch. Its score (340)
beats the runner-up (240) by a clear margin because the synthetic charge pattern is
*chiral*, so no rotated alternative matches it. The `PASS` line confirms the GPU
and CPU agree on all 648 pose scores, bit-for-bit.

The stderr timing line (not shown above, since it varies) reports the CPU and GPU
kernel times. On this tiny sample both are well under a millisecond and dominated
by launch/copy overhead — the GPU's advantage only shows once the pose space and
atom counts grow (see THEORY §4).
