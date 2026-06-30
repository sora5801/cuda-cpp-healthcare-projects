# Demo — 2.18 NMR Structure Refinement

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/restraints.txt` input.
3. **Verify** the GPU ensemble against the CPU reference (`reference_cpu.cpp`) and
   print a clear `PASS`/`FAIL`.
4. **Time** the kernel (CUDA events) and the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric error (which vary run to run), so it
  is shown but never diffed.

## What you are looking at

The demo anneals **512 independent simulated-annealing trajectories** (one GPU thread
each) against 19 NOE distance restraints generated from a known α-helix. Each replica
starts from a different random chain and cools from T = 5.0 to 0.02. The output
reports:

- the **best replica** — the lowest restraint-energy structure found across the
  ensemble (the "NMR structure" you would publish);
- five **sample replicas** so you can see the spread of outcomes;
- the **ensemble summary** — how many of the 512 replicas satisfied *all* restraints;
- the **PASS/FAIL** line confirming the GPU ensemble matches the CPU reference
  (integer counts exactly; energy within `1e-4`).

That the best replica reaches ~0.03 energy with **19/19** restraints satisfied is the
science working: the annealer recovered a structure consistent with the (synthetic)
NOE data.

## Expected result

```
2.18 -- NMR Structure Refinement
ensemble SA: 512 replicas x 4000 steps; chain=12 beads, 19 NOE restraints
schedule: T 5.00 -> 0.02 (geometric), trial sigma=1.20 A, bond=3.80 A
best replica: #1  energy=0.0328  restraints satisfied=19/19
sample replicas (idx -> energy satisfied):
  r0   :    0.1392  19/19
  r128 :    0.4354  18/19
  r256 :    0.2986  19/19
  r384 :    0.3911  17/19
  r511 :    0.3584  19/19
ensemble: 298/512 replicas satisfy all restraints; mean best energy=0.2954; max=1.0230
RESULT: PASS (GPU ensemble matches CPU: counts exact, energy within 1.0e-04)
```

The stderr timing line (e.g. `[timing] CPU: 379 ms   GPU kernel: 182 ms`) will differ
on your machine — that is expected and is why it is not part of the diffed output.
