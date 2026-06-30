# Demo — 2.34 Biophysical Simulation of Biomolecular Condensates (Active Learning Loop)

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/condensate_ensemble.txt` input.
3. **Verify** the GPU ensemble against the CPU reference (`reference_cpu.cpp`),
   per replica, and print a clear `PASS`/`FAIL`.
4. **Time** the GPU kernel (CUDA events) and the CPU baseline — a *teaching
   artifact*, not a benchmark claim.

The program splits its output deliberately (PATTERNS.md §3):

- **stdout** is byte-for-byte deterministic (the counter-based RNG makes every
  trajectory reproducible) and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the worst CPU↔GPU difference (which vary run
  to run), so it is shown but never diffed.

## Expected result

```
2.34 -- Biophysical Simulation of Biomolecular Condensates (Active Learning Loop)
reduced-scope teaching model: coarse-grained Brownian-dynamics condensate ensemble
ensemble: 24 candidate sequences (stickiness lambda in [0.50, 8.00])
CG-MD: 12 beads, 500 steps (dt=0.005, eq=150), kT=1.00, target D=0.16500
sample replicas (lambda -> Rg  D  |D-target|):
  m0    lambda=0.500 -> Rg=2.90423  D=0.19940  |dD|=0.03440
  m6    lambda=2.457 -> Rg=1.90124  D=0.17254  |dD|=0.00754
  m12   lambda=4.413 -> Rg=1.34708  D=0.16357  |dD|=0.00143
  m18   lambda=6.370 -> Rg=1.12106  D=0.15870  |dD|=0.00630
  m23   lambda=8.000 -> Rg=0.94504  D=0.15994  |dD|=0.00506
active-learning proposal: member m12, lambda=4.413 (D=0.16357 closest to target 0.16500)
RESULT: PASS (GPU ensemble matches CPU within tol=1.0e-06)
```

## How to read it

- Each **replica** is one candidate IDP sequence reduced to a single "stickiness"
  `lambda`. The table prints five evenly-spaced members.
- **`Rg`** (radius of gyration) falls monotonically as `lambda` rises: stickier
  sequences fold up more tightly — the condensate-compaction trend the model
  teaches.
- **`D`** is the internal mobility (diffusion of beads relative to the chain's
  centre of mass) from a lag-MSD Einstein relation; it trends downward with
  `lambda` but carries visible thermal noise (single short trajectory) — that
  noise is honest, not a bug.
- The **active-learning proposal** is the headline of the loop: the member whose
  measured `D` is closest to the experimental `target D = 0.165`. Here member
  `m12` (`lambda = 4.413`, `D = 0.16357`) wins — the sequence Bayesian
  optimization would simulate at higher fidelity next.
- The CPU and GPU agree to ~`1e-15` (the stderr `worst per-replica diff`),
  comfortably inside the `1e-6` tolerance.

> On a tiny 24-member ensemble the GPU is *slower* than the CPU (it is launch- and
> per-trajectory-local-memory-bound); the GPU's advantage appears as the ensemble
> grows to the hundreds of replicas a real active-learning iteration uses. This is
> the honest-timing rule (PATTERNS.md §7).
