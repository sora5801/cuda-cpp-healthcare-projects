# Demo — Project 6.18 ECG Forward Problem & Body-Surface Potential Mapping

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/ecg_sample.txt` (a synthetic torso).
3. **Verify** the GPU result against the CPU reference (`reference_cpu.cpp`): the
   GPU-built lead-field matrix and the GPU-computed body-surface potentials must
   match the CPU within a documented tolerance — prints `PASS`/`FAIL`.
4. **Time** the GPU lead-field kernel and the cuBLAS DGEMM (CUDA events) plus the
   CPU baseline — a *teaching artifact*, not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric error (which vary run to run), so
  it is shown but never diffed.

## What you are looking at

- **per-lead peak-to-peak** — for each of the `L` electrodes, the max-minus-min of
  its predicted body-surface potential trace over the `T` frames. This is the
  clinically-familiar "how big is this lead's deflection" summary.
- **largest-swing lead** — which electrode swings the most. The synthetic sample
  was built so the electrode nearest the strongest cardiac source (electrode 0)
  *must* win; the demo reports `RECOVERED` when the computed answer matches that
  geometric ground truth (the human-meaningful headline).
- **signature Phi[lead 0][last frame]** — one fixed potential value, so any change
  to the physics or the matrix multiply shows up as a changed digit.

## Expected result

The committed [`expected_output.txt`](expected_output.txt) is captured from a real
run on the sample. It looks like:

```
6.18 -- ECG Forward Problem & Body-Surface Potential Mapping
torso model: L=8 electrodes, S=3 dipole sources, T=24 frames (synthetic)
conductivity sigma=0.2000 S/m (homogeneous volume conductor)
per-lead peak-to-peak body-surface potential (lead: p2p):
  lead  0: p2p=...
  ...
largest-swing lead: 0  (expected from geometry: 0) -> RECOVERED
signature Phi[lead 0][frame 23] = ...
RESULT: PASS (GPU lead field and potentials match CPU within tol)
```

(The exact numbers are in `expected_output.txt`.) The `stderr` timing line varies
per machine and is not part of the diff.
