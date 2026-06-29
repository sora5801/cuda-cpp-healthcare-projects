# Demo — 1.32 Alchemical Hydration Free Energy (ΔGsolv)

## What this demonstrates

`run_demo.ps1` (Windows) / `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on `data/sample/alchemy_config.txt` — 11 λ-windows × 64 Monte Carlo
   walkers = **704 independent chains** (one GPU thread each), 1000 MC steps per
   chain, sampling a single Lennard-Jones solute being alchemically coupled into
   a fixed solvent bath.
3. **Verify** the GPU per-walker accumulators match the CPU reference (both run
   the *identical* `run_walker()` from `src/alchemy.h`).
4. **Report** the per-window TI integrand `⟨∂U/∂λ⟩`, the acceptance ratio, and the
   solvation free energy ΔG_solv computed two independent ways — **Thermodynamic
   Integration** (trapezoid over λ) and **BAR** (Bennett acceptance ratio over
   adjacent windows).

stdout (the deterministic table + ΔG) is diffed against
[`expected_output.txt`](expected_output.txt); the timing line is on stderr only
(it varies run to run and is a teaching artifact, never a benchmark claim).

## Canonical output

See [`expected_output.txt`](expected_output.txt). What to notice, and why it is
the physics behaving correctly (not just two codes agreeing):

- **`⟨∂U/∂λ⟩` falls monotonically** from ~+0.2 at λ=0 toward ~−1.6 at λ=1. As the
  solute–solvent interaction switches on, the favorable Lennard-Jones attraction
  dominates, so coupling it in lowers the energy — exactly the sign TI integrates.
- **Acceptance drops** from ~91% (λ=0, the solute is nearly a ghost, every move is
  accepted) to ~64% (λ=1, the solute now feels real walls and wells) — a textbook
  Monte Carlo signature of a stiffer landscape.
- **TI and BAR agree** to ~0.04 (reduced units): two estimators with different
  systematic errors landing on the same ΔG is the standard self-consistency check
  in free-energy work.
- **`RESULT: PASS`** means the GPU and CPU per-walker sums agree to ≈1.5e−11, far
  inside the documented 1e−9 tolerance — the residual is double-precision FMA
  reordering between nvcc and the host compiler (THEORY §5), not an algorithm
  difference.

> **Reduced LJ units, synthetic model.** The ΔG here is a correct TI/BAR result
> for a toy single-particle solute, **not** a force-field prediction of any real
> molecule's hydration free energy. See `data/README.md` and the README
> "Limitations".
