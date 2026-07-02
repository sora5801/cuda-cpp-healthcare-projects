# Demo — 6.17 Purkinje System & Conduction System Modeling

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/purkinje_tree.txt` — a tiny synthetic
   His-Purkinje tree of 7 cables.
3. **Verify** the GPU result against the CPU reference (`reference_cpu.cpp`): the
   per-cable activation-step indices must match **exactly** (they are integers)
   and the conduction velocities to within `1e-9` mm/ms. Prints `PASS`/`FAIL`.
4. **Time** the kernel (CUDA events) and the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the verify diagnostics (which vary run to
  run), so it is shown but never diffed.

## How to read the output

Each row is one Purkinje cable:

```
c<idx> p<parent> <length_mm> <D> -> <CV mm/ms> <PMJ activation time ms> <captured?>
```

- **`D`** is the diffusion (axial) coefficient — a proxy for fibre diameter.
  Larger `D` → faster **conduction velocity (CV)**. Compare cable `c1` (D=3.0,
  CV≈2.59) against `c2` (D=1.5, CV≈1.80): halving `D` slows the wave, exactly the
  His-Purkinje CV-vs-diameter relationship clinicians calibrate.
- **PMJ activation time** is the absolute time (ms) at which each cable's distal
  Purkinje-muscle junction fires, assembled by the O(N) graph-delay pass over the
  tree (root → branches → leaves).
- **total ventricular activation** (~30 ms here) is the latest PMJ time — the
  clinically meaningful "how long to activate the whole tree" number.

## Expected result

```
6.17 -- Purkinje System & Conduction System Modeling
Purkinje tree: 7 cables, dt=0.010 ms, 6000 steps (60.0 ms)
per-cable (idx parent lenmm D -> CV[mm/ms] PMJ_t[ms] captured):
  c0  p-1   20.0 3.00 ->  2.6247    7.620 yes
  c1  p0    25.0 3.00 ->  2.5853   18.290 yes
  c2  p0    25.0 1.50 ->  1.7973   22.530 yes
  c3  p1    15.0 2.50 ->  2.4430   24.930 yes
  c4  p1    15.0 2.50 ->  2.4430   24.930 yes
  c5  p2    15.0 2.00 ->  2.1614   29.970 yes
  c6  p2    15.0 2.00 ->  2.1614   29.970 yes
tree: 7/7 cables captured; total ventricular activation = 29.970 ms
RESULT: PASS (GPU per-cable steps + CV match CPU; tol=1.0e-09)
```

Timing (on **stderr**, varies) shows the CPU baseline and the GPU kernel time.
On this tiny 7-cable tree the GPU is *slower* — launch overhead dominates when
there are only 7 threads of work. The GPU wins once the ensemble grows toward the
tens of thousands of segments in a real Purkinje network (see THEORY §7 honesty).

> All data here is **synthetic** and for teaching only — never clinical use.
