# Demo — 1.25 Gaussian-Accelerated MD (GaMD)

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** an ensemble of 512 GaMD-boosted Langevin walkers on the committed
   `data/sample/gamd_config.txt` — once on the **CPU** (serial reference) and once
   on the **GPU** (one thread per walker).
3. **Verify** two things:
   - the GPU fixed-point histogram tally equals the CPU tally **bit-for-bit**
     (integer atomics are order-independent, so the tolerance is exactly **0**); and
   - the reweighted free-energy profile (PMF) recovers the **known** double-well
     barrier height (a *scientific* check, to a documented physical tolerance).
4. **Time** the kernel (CUDA events) vs. the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the verify diagnostics (which vary run to run),
  so it is shown but never diffed.

## How to read the output

The headline is the **reweighted PMF table**: for nine evenly spaced positions `x`,
it prints `F_sim` (the free energy GaMD reconstructed from the *boosted* run) next
to `F_true` (the analytic double well `U(x)`, minimum shifted to 0). A correct
GaMD run makes `F_sim` track `F_true`: ~0 at the two wells (`x≈±1`) and rising to
the barrier near `x≈0`. The final lines report:

- **recovered barrier height** vs. the true value — the science check; and
- **enhanced sampling** — how many bins in each well were visited, showing the
  boost let walkers populate **both** wells (a low-temperature *unboosted* run
  would be trapped on one side).

## Expected result

```
1.25 -- Gaussian-Accelerated MD (GaMD)
model: double-well U(x)=3.0*(x^2-1)^2  kT=1.00  walkers=512  steps=8000 (equil 2000)
GaMD boost: E=3.00  k0=0.15  k=0.0500  (dV=0.5*k*(E-U)^2 for U<E)
reweighted PMF (2nd-order cumulant), F(x) in kT, min shifted to 0:
  x=-1.95 : F_sim=  n/a  F_true= 23.56
  x=-1.55 : F_sim=  5.44  F_true=  5.90
  x=-1.05 : F_sim=  0.03  F_true=  0.03
  x=-0.55 : F_sim=  1.62  F_true=  1.46
  x=-0.05 : F_sim=  3.27  F_true=  2.99
  x=+0.45 : F_sim=  2.15  F_true=  1.91
  x=+0.95 : F_sim=  0.03  F_true=  0.03
  x=+1.45 : F_sim=  3.66  F_true=  3.65
  x=+1.95 : F_sim=  n/a  F_true= 23.56
recovered barrier height = 3.26 kT  (true = 3.00 kT)
enhanced sampling: 18 left-well + 18 right-well bins visited (both wells: yes)
RESULT: PASS (GPU tally == CPU exactly; barrier recovered; both wells sampled)
```

The `n/a` rows are bins outside the sampled region (`|x|>~1.8`), where the
potential is so high that no walker visited them in a short run — expected and
honest. The recovered barrier (3.26 kT) is slightly **above** the true 3.00 kT;
that residual is the 2nd-order cumulant truncation bias discussed in
[`../THEORY.md`](../THEORY.md) §6 — a real, taught property of GaMD, not a bug.
