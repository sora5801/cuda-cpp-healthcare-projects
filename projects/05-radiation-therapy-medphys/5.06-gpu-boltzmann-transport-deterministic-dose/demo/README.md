# Demo — 5.6 GPU Boltzmann Transport (Deterministic Dose)

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/slab_problem.txt` — a synthetic
   layered "tissue / lung / tissue" slab with a source band.
3. **Solve** the 1-D discrete-ordinates (S₈) transport problem by source
   iteration on both the CPU reference and the GPU, and **verify** the GPU scalar
   flux matches the CPU flux, printing a clear `PASS`/`FAIL`.
4. **Time** the iteration loop (CUDA events) and the CPU baseline — a *teaching
   artifact*, not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic (both solvers are deterministic and
  reduce ordinate contributions in a fixed order) and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing, the GPU-vs-CPU error, and a physics sanity line
  (all run-varying or informational), so it is shown but never diffed.

## Expected result

```
5.6 -- GPU Boltzmann Transport (Deterministic Dose)
S_8 discrete ordinates, 24 cells over 6.000 cm (h=0.2500 cm), tol=1.0e-10
source iterations: CPU=66 GPU=66
  cell        x_cm         scalar_flux         dose_proxy
     0       0.1250         4.070628e-01       8.141256e-02
     ...
    23       5.8750         3.576380e-02       7.152760e-03
peak scalar flux = 9.518772e-01 at cell 3 (x=0.8750 cm)
RESULT: PASS (GPU flux matches CPU within tol=1.0e-11)
```

## What to look for (the teaching payoff)

- **The flux profile.** `scalar_flux` peaks in the source band (cells 2–3, where
  `q>0`) and decays with distance, dropping to zero-ish at the vacuum faces where
  particles leak out. That shape is the deterministic transport solution — no
  Monte-Carlo noise, so it is smooth to machine precision.
- **The tissue↔lung contrast.** `dose_proxy = Sigma_a * phi` drops sharply in the
  low-density "lung" middle third (cells 8–15): the flux there is still
  substantial, but with tiny absorption (`Sigma_a = 0.05`) little energy is
  deposited. This heterogeneity — high flux, low deposition — is exactly why
  deterministic LBTE engines (Acuros XB) were built to replace pencil-beam
  superposition in lung.
- **GPU == CPU to ~5e-17** (stderr `[verify]` line): the shared `__host__
  __device__` physics and fixed-order angular reduction make the two paths agree
  to round-off, so the tolerance (1e-11) is met with enormous margin.
- **The physics sanity line** (stderr `[physics]`): the source-cell flux sits
  well below the infinite-medium value `phi_inf = q/Sigma_a`, the signature of a
  finite slab that leaks at its boundaries.
