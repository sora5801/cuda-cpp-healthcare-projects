# Demo — 1.35 QMMM/ML Potential Hybrid MD

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/ensemble_params.txt` config.
3. **Verify** the GPU ensemble against the CPU reference (`reference_cpu.cpp`) and
   print a clear `PASS`/`FAIL`.
4. **Time** the kernel (CUDA events) and the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

Each "ensemble member" is one short **hybrid NNP/MM molecular-dynamics trajectory**:
a 1-D chain whose reactive center is described by a (surrogate) neural-network
potential and whose environment is classical Lennard-Jones, coupled across a
link-atom boundary by mechanical embedding. The 64 members differ only by a fixed
perturbation of the link atom — an *active-learning*-style sweep of configuration
space. One GPU thread runs one whole trajectory (the **ensemble** pattern).

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the verification error (which vary run to
  run), so it is shown but never diffed.

## Expected result

```
1.35 -- QMMM/ML Potential Hybrid MD
REDUCED-SCOPE TEACHING VERSION: NNP weights are fixed/synthetic.
hybrid NNP/MM chain: 8 atoms (4 MM, link@4, 4 ML), NNP=4x3 MLP
ensemble: 64 trajectories, 300 steps @ dt=0.005, link perturbation +/-0.200
sample members (idx perturb -> finalPE finalE maxForce):
  m0    -0.200 ->    -2.480797     0.705637     1.924769
  m16   -0.098 ->    -4.456266    -4.060512     5.934750
  m32   +0.003 ->    -4.755624    -4.590117     0.267700
  m48   +0.105 ->    -4.639954    -4.363458     3.268006
  m63   +0.200 ->    -4.353222    -4.058608     5.792803
unperturbed (m32): finalPE=-4.755624  finalE=-4.590117
worst energy conservation (max |finalE - initialE|) = 0.199177
RESULT: PASS (GPU ensemble matches CPU within tol=1.0e-06)
```

### How to read it

- **finalPE / finalE** — potential and total (PE + KE) energy at the end of the
  trajectory, in the model's arbitrary energy units.
- **maxForce** — the largest force magnitude on any atom at the last step; a gauge
  of how "stiff" that configuration is.
- **worst energy conservation** — the largest `|finalE − initialE|` over all 64
  members. Velocity-Verlet is *symplectic*, so total energy stays **bounded** (it
  does not drift away secularly). The worst case here (`m0`, the link atom shoved
  −0.2 into the Lennard-Jones repulsive wall) is the stiffest configuration; the
  near-equilibrium members conserve energy far better. That is a real lesson:
  **stiff regions need a smaller timestep** (try `--dt` in `make_synthetic.py`).
- **RESULT** — the GPU per-member summaries match the CPU reference to
  `3.99e-13`, well inside the documented `1e-6` tolerance (see THEORY §verify).

> **Reduced-scope teaching version.** The NNP weights are fixed synthetic
> surrogates standing in for a model trained on QM data (MACE/NequIP). The
> energies are not physical quantities for any real molecule. Educational only —
> not for any clinical or chemical decision.
