# Demo — 4.7 Medical Image Segmentation (Deep Learning)

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed synthetic volume in `data/sample/`.
3. **Segment** the volume with a tiny 2-layer 3D-convolutional head, on both the
   CPU reference and the GPU, and **verify** they agree: the integer label map
   must be **identical** and the lesion logits must match within `1e-3`.
4. **Score** the predicted mask against the known ground-truth lesion sphere with
   the **Dice coefficient** (≈ 0.96), and print the central-slice mask.
5. **Time** the two kernels (CUDA events) vs. the CPU baseline — a *teaching
   artifact*, not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic (voxel counts, Dice, the ASCII mask)
  and is diffed against [`expected_output.txt`](expected_output.txt). It is
  deterministic because the label map and the Dice are computed from **integer**
  counts (no order-dependent float reduction).
- **stderr** carries the timing and the float error (which vary run to run), so
  it is shown but never diffed.

## Expected result

```
4.7 -- Medical Image Segmentation (Deep Learning)  [reduced-scope teaching version]
volume: 12x16x16 (3072 voxels), 2-layer 3D conv head, 2 classes
predicted lesion voxels = 133
ground-truth lesion voxels = 123
Dice(prediction, ground truth) = 0.9609
predicted mask, central z=6 slice (1=lesion, .=bg):
  ................
  ................
  ................
  ................
  .........1......
  .......11111....
  .......111111...
  ......1111111...
  .......111111...
  .......11111....
  ........111.....
  ................
  ................
  ................
  ................
  ................
RESULT: PASS (GPU label map == CPU exactly; logits within tol=1.0e-03)
```

The circular cross-section in the central slice is the lesion sphere — the
network recovered it from the noisy intensity volume. A Dice of 0.96 means the
predicted mask and the ground-truth sphere overlap almost perfectly.

## Reading the verification

- `[verify] label mismatches = 0` — the GPU and CPU label **every** voxel
  identically (the exact-integer gate).
- `max logit err ≈ 1e-7` — the continuous lesion logits also match to well within
  the `1e-3` tolerance; they are not bit-identical because the GPU fuses
  multiply-adds (FMA) where the host may not. See `THEORY.md` §"Numerical
  considerations".

## Regenerating `expected_output.txt`

If you change the kernel, the weights, or the sample, regenerate it from a **real
run** (never hand-edit it):

```powershell
$dir = (Resolve-Path "$PSScriptRoot\..").Path
& "$dir\build\x64\Release\medical-image-segmentation-deep-learning.exe" `
    "$dir\data\sample\volume_sample.txt" 1>"$dir\demo\expected_output.txt"
```
