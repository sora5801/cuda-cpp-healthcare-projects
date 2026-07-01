# Demo — 5.3 Proton & Heavy-Ion Therapy Dose

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/proton_plan_sample.txt` plan (a single
   synthetic proton pencil beam of range 12 cm).
3. **Verify** the GPU dose volume against the CPU reference (`reference_cpu.cpp`)
   voxel-by-voxel, and confirm the **Bragg-peak depth bin agrees**; prints
   `PASS`/`FAIL`.
4. **Time** the kernel (CUDA events) and the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

The printed table is the **integral depth-dose** (dose summed over the lateral
plane at each depth slice). It shows the hallmark of proton therapy: a low
entrance plateau (~0.27 of peak), a sharp **Bragg peak** just proximal to the
12 cm range, then a **hard fall to zero** distal to the range — the physics that
lets protons spare tissue behind the target. The ASCII bars make the peak visible
at a glance.

The program splits its output deliberately (docs/PATTERNS.md §3):

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries timing and the numeric error (which vary run to run), so it
  is shown but never diffed.

## Expected result (stdout)

```
5.3 -- Proton & Heavy-Ion Therapy Dose
[teaching model: analytic pencil-beam dose; arbitrary units; not clinical]
grid = 9x9x40 voxels @ 0.50 cm, entry z = 0.00 cm, spots = 1
integral depth-dose (lateral sum per slice, normalized to Bragg peak):
  z= 0.25 cm  0.269114  ###########
  ...
  z=11.25 cm  0.783894  ###############################
  z=11.75 cm  1.000000  ########################################
  z=12.25 cm  0.000000
  ...
Bragg-peak depth = 11.75 cm (bin 23)
RESULT: PASS (GPU dose matches CPU within tol=1.0e-04; peak bins agree)
```

The full curve is in [`expected_output.txt`](expected_output.txt); the demo diffs
against every line of it. Note the peak lands at **11.75 cm**, half a voxel
proximal to the input **12 cm** range — exactly where the Bragg peak sits.

## Reading the verification

- `max_abs_err` (stderr) is the largest per-voxel |GPU − CPU| difference; it is
  ~`1.8e-7` (a few FP32 ULPs) because the CPU and GPU evaluate the *same* formula
  (`src/proton_physics.h`) in the same precision and summation order.
- The Bragg-peak bin must be identical on both sides — a science-level check that
  the depth shape (not just the arithmetic) matches.
