# Demo — 1.30 Trajectory RMSD, Clustering & Contact Analysis

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/trajectory_sample.txt`.
3. **Verify** the GPU result against the CPU reference (`reference_cpu.cpp`) and
   print a clear `PASS`/`FAIL`.
4. **Time** the kernel (CUDA events) and the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric error (which vary run to run), so
  it is shown but never diffed.

## How to read the output

- The **per-frame table** gives, for each frame, its optimal-superposition
  **RMSD** to the reference (frame 0) and its **native-contact fraction Q**. Watch
  RMSD climb from `0.0000` and `Q` fall from `1.0000` as the synthetic helix
  unfolds — frame 0 matches the reference exactly (RMSD 0, Q 1), the analytic
  anchor that proves the pipeline is correct.
- The **RMSD clustering** block bins frames into shells of width 1.0. Three shells
  are populated (≈0, ≈5, ≈10) with empty shells between — the three metastable
  states the synthetic trajectory was built to visit. This is the didactic
  stand-in for true pairwise-RMSD GROMOS clustering (see THEORY "real world").
- The **stderr** line `max_abs_err: rmsd=… Q=…` is the GPU-vs-CPU agreement: both
  run the same FP64 `rmsd_core.h` math, so the error is ~`5e-14` (≪ the `1e-9`
  tolerance), and `Q` matches **exactly**.

## Expected result (stdout)

```
1.30 -- Trajectory RMSD, Clustering & Contact Analysis
frames=12  atoms=16  reference=frame 0
per-frame RMSD (to reference, optimal superposition) and Q (native contacts):
  frame  0   RMSD =   0.0000   Q = 1.0000
  frame  1   RMSD =   0.4531   Q = 1.0000
  frame  2   RMSD =   0.9061   Q = 1.0000
  frame  3   RMSD =   1.1326   Q = 1.0000
  frame  4   RMSD =   4.7572   Q = 0.5200
  frame  5   RMSD =   5.0970   Q = 0.5200
  frame  6   RMSD =   5.3235   Q = 0.0000
  frame  7   RMSD =   5.5500   Q = 0.0000
  frame  8   RMSD =   9.7410   Q = 0.0000
  frame  9   RMSD =  10.0808   Q = 0.0000
  frame 10   RMSD =  10.4207   Q = 0.0000
  frame 11   RMSD =  10.7605   Q = 0.0000
RMSD clustering (shell width = 1.0):
  shell 0  [0.0, 1.0):  3 frame(s)
  shell 1  [1.0, 2.0):  1 frame(s)
  shell 2  [2.0, 3.0):  0 frame(s)
  shell 3  [3.0, 4.0):  0 frame(s)
  shell 4  [4.0, 5.0):  1 frame(s)
  shell 5  [5.0, 6.0):  3 frame(s)
  shell 6  [6.0, 7.0):  0 frame(s)
  shell 7  [7.0, 8.0):  0 frame(s)
  shell 8  [8.0, 9.0):  0 frame(s)
  shell 9  [9.0, 10.0):  1 frame(s)
  shell 10  [10.0, 11.0):  3 frame(s)
RESULT: PASS (GPU matches CPU within tol=1.0e-09)
```

The matching `stderr` (timing + error) is shown by the demo but not diffed, e.g.:

```
[verify] max_abs_err: rmsd=5.507e-14  Q=0.000e+00  (tolerance 1.0e-09)
```
