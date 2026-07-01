# Demo — 6.1 Cardiac Electrophysiology Simulation

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/tissue_params.txt` input — a 32×32
   sheet of excitable tissue sparked by an S1 stimulus on the left edge.
3. **Verify** the GPU voltage field against the CPU reference
   (`reference_cpu.cpp`) and print a clear `PASS`/`FAIL`.
4. **Time** the kernel loop (CUDA events) and the CPU baseline — a *teaching
   artifact*, not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric error (which vary run to run), so
  it is shown but never diffed.

## Reading the result

The headline line is the **voltage slice** `V(x, y=16)` — a horizontal cut
through the middle row of the tissue. Because the S1 stimulus fires on the LEFT
edge, an action-potential wave travels rightward. In the slice you can read the
wave off directly:

- Values near **1.0** on the left = tissue the wave has already depolarised.
- The sharp drop toward **0.0** on the right = the **wavefront** (the leading
  edge of the electrical wave) and the resting tissue ahead of it.

`activated(V>0.5)` counts how many cells are currently depolarised — a proxy for
how far the wave has spread. `RESULT: PASS` means the GPU field matches the CPU
field within `1e-6` (they run the *same* double-precision physics via the shared
`cardiac_cell.h`; the tiny residual is FMA rounding — see THEORY §Numerics).

## Expected result

The exact deterministic stdout is captured in
[`expected_output.txt`](expected_output.txt) (generated from a real run, never
hand-written). It begins:

```
6.1 -- Cardiac Electrophysiology Simulation
monodomain (FitzHugh-Nagumo reaction + diffusion), operator split
grid 32x32, 400 steps, dt=0.1000 dx=1.000 D=0.2000 (CFL dt_max=1.2500)
FHN: a=0.100 eps=0.0020 b=0.500 | S1 patch 3x32 at (0,0) V=1.00
final V: min=0.000000 max=0.963826 | activated(V>0.5)=384 (37.5%)
```

and ends with `RESULT: PASS`. The `voltage slice` line shows the depolarised
plateau (~0.90–0.96) on the left, the sharp wavefront near `x≈11`, and resting
tissue (`0.0000`) ahead of the wave.
