# Demo — 1.26 Steered Molecular Dynamics (SMD)

## What this demonstrates

`run_demo.ps1` (Windows) / `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on `data/sample/smd_config.txt` — 8192 independent constant-velocity
   SMD pulls, each 25000 overdamped-Langevin steps, dragging a reaction
   coordinate from the bound state (0 nm) to the unbound state (1 nm).
3. **Verify** two things:
   - the GPU's per-trajectory external work matches the CPU reference (the two
     share the same RNG and integrator, so they agree to ~1e-12 kJ/mol), and
   - **Jarzynski's equality** recovers the *known* free-energy difference of the
     model's potential of mean force within a documented tolerance.
4. **Report** a few sample work values, the work distribution, and three
   free-energy numbers side by side.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the GPU-vs-CPU error (which vary run to run),
  so it is shown but never diffed.

## Expected result

See [`expected_output.txt`](expected_output.txt). The headline is the contrast in
the `free energy` line:

```
free energy (kJ/mol): naive <W>=-1.8871  Jarzynski dG=-11.3200  true dG=-12.0000
```

- **naive ⟨W⟩ = −1.89** — just averaging the pulling work. The second law forces
  `⟨W⟩ ≥ ΔG`, so this is a badly biased estimate; the `dissipation <W>-dG`
  line (≈ 10 kJ/mol) is exactly that bias, the energy lost to friction.
- **Jarzynski ΔG = −11.32** — the *exponential* average
  `−kT·ln⟨exp(−W/kT)⟩`. It up-weights the rare low-work trajectories that
  dominate the free energy and lands within ~0.7 kJ/mol of the truth, **from
  non-equilibrium pulls**. That is the whole point of the method.
- **true ΔG = −12.00** — the analytic free-energy difference of the engineered
  potential of mean force (`pmf_slope · 1 nm`), the ground truth.

`RESULT: PASS` means both checks held. On the sample the GPU runs the 8192
trajectories ~20–30× faster than the single-threaded CPU reference — a gap that
grows with the ensemble size, which is exactly the lever Jarzynski estimates need
(more trajectories → better-sampled tail → tighter ΔG).

> The model is a **synthetic, reduced 1-D caricature** of full-atom SMD — a
> software demonstration of the *method* (pull, accumulate work, apply
> Jarzynski), not a simulation of any real molecule and not for any design or
> clinical use.
