# Demo — 1.11 QSAR / Property Prediction

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/molecules_sample.txt` — a batch of 5
   tiny synthetic "molecules".
3. **Predict a property** for every molecule by running a 2-layer **Graph
   Convolutional Network** (message-passing) plus a mean-pool readout, both on the
   GPU (`kernels.cu`) and on the CPU reference (`reference_cpu.cpp`).
4. **Verify** the GPU predictions against the CPU ones and print `PASS`/`FAIL`.
5. **Time** the kernels (CUDA events) and the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing, the data path, and the numeric error (which vary
  run to run / machine to machine), so it is shown but never diffed.

## Expected result (stdout)

```
1.11 -- QSAR / Property Prediction
GCN inference: 5 molecules, 23 atoms total (F_IN=6, F_HID=8, F_OUT=4)
  mol  0 ( 4 atoms): predicted property = -0.582784
  mol  1 ( 6 atoms): predicted property = -0.651793
  mol  2 ( 4 atoms): predicted property = -0.554825
  mol  3 ( 4 atoms): predicted property = -0.536602
  mol  4 ( 5 atoms): predicted property = -0.641983
top-ranked molecule: mol 3 (property = -0.536602)
RESULT: PASS (GPU predictions match CPU within 1e-04)
```

## How to read it

- Each line is **one molecule's** predicted property — the scalar a QSAR model
  outputs. The **top-ranked** molecule (highest score) is the one a virtual screen
  would shortlist; here it is `mol 3`, the branched C(C,N,O) star.
- `RESULT: PASS` means the GPU and CPU predictions agree to within `1e-4`. They run
  the *same* `gcn.h` math in the *same* neighbor order, so the real gap (printed on
  stderr) is ~`6e-8` — pure fp32 fused-multiply-add rounding (see THEORY §6).

> **Honesty:** the model weights are **untrained** (a seeded generator), so these
> numbers carry no chemical meaning — they demonstrate the GPU message-passing
> pipeline and the exact CPU↔GPU agreement, nothing more. Not for any real use.
