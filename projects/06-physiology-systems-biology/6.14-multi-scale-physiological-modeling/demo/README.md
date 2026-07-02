# Demo — 6.14 Multi-Scale Physiological Modeling

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on `data/sample/cable.txt` — a 128-node 1-D cardiac cable with a
   FitzHugh-Nagumo (FHN) cell model at every node, coupled by tissue diffusion.
3. **Verify** that the GPU result (final `v`/`w` fields + activation map) matches
   the serial CPU reference (`reference_cpu.cpp`), and print `PASS`/`FAIL`.
4. **Time** the GPU kernels (CUDA events) and the CPU baseline — a *teaching
   artifact*, not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric field-error (which vary run to
  run), so it is shown but never diffed.

## Expected result

See [`expected_output.txt`](expected_output.txt):

```
6.14 -- Multi-Scale Physiological Modeling
1-D monodomain cable: 128 nodes, dx=0.500, dt=0.0200, 5000 steps (T=100.00)
FHN cell: a=0.130 eps=0.005 b=0.500 | tissue D=2.000 | stim 5 left nodes
activation map (node : x : t_activation):
  n0       0.000    0.0000
  n25     12.500   17.6200
  n50     25.000   36.5600
  n76     38.000   56.2600
  n101    50.500   75.2200
  n127    63.500   90.2000
nodes activated: 128 / 128
conduction velocity: 0.6790 (space/time)
RESULT: PASS (GPU field matches CPU within tol=1.0e-06)
```

## How to read it

The **activation map** is the heart of the result: the time at which each node's
voltage first crosses threshold *increases* with position (0.0 → 17.6 → 36.6 →
… → 90.2 as you move down the cable). That monotone increase **is** the action
potential propagating — a traveling wave born at the stimulated left end and
sweeping to the far end (all 128 nodes activate). Its slope gives the
**conduction velocity** (~0.68 space/time), the physiological headline number:
the cardiac analogue of how fast electrical activation spreads across tissue.

`RESULT: PASS` means the GPU and CPU fields agree to within `1e-6`; in practice
they agree to `~1e-16` (the operator-split arithmetic is identical on both
sides — see THEORY.md "How we verify correctness").

> The parameters and units are **illustrative** (a didactic FHN reduction), not
> fitted to any real heart — a software demonstration of multi-scale coupling,
> not a clinical simulation.
