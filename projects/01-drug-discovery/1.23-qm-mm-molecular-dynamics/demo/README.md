# Demo — 1.23 QM/MM Molecular Dynamics

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on `data/sample/ensemble_params.txt` — a 16×16 sweep = **256
   QM/MM trajectories** of a model proton transfer.
3. **Verify** that the GPU ensemble matches the CPU reference (both call the same
   `__host__ __device__` QM/MM core in `src/qmmm.h`), printing `PASS`/`FAIL`.
4. **Report** five sample trajectories and an ensemble summary, then time the
   kernel (CUDA events) and the CPU baseline — a *teaching artifact*, not a
   benchmark claim.

The program splits its output deliberately (PATTERNS.md §3):

- **stdout** is byte-for-byte deterministic (fixed precision) and is diffed
  against [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the worst-case verification error (which vary
  run to run), so it is shown but never diffed.

## How to read the output

Each sampled line is `field x0 -> final_x final_E min_gap %product transferred`:

- **field** — the MM electrostatic-embedding bias for that trajectory. `0` means
  a symmetric double well; more negative tilts the surface toward the acceptor.
- **x0** — initial proton position (both sample values sit inside the *donor*
  well at `x < 0`).
- **final_x** — where the proton ends. `> 0` means it crossed to the acceptor.
- **final_E** — total QM/MM potential energy at the end.
- **min_gap** — the smallest adiabatic gap (ground↔excited state splitting) seen
  along the way; it bottoms out at `2·coupling = 4.0` when the proton sits over
  the barrier (the diabatic states are degenerate there). A small gap is where a
  real simulation would worry about non-adiabatic (surface-hopping) effects.
- **%product / transferred** — fraction of steps spent on the acceptor side, and
  whether the run ended there.

## Canonical output

See [`expected_output.txt`](expected_output.txt). The physics is textbook: at
`field = 0` (trajectory `t0`) the proton stays **trapped** in the donor well
(`transferred 0`, `%product 0.00`, large `min_gap` because it never approaches the
barrier). As the embedding field grows more negative, the surface tilts, the
proton gains energy, crosses the barrier, and **transfers** — exactly how a
charged residue or solvent dipole drives proton transfer in an enzyme. Over the
sweep, **90 / 256** trajectories transfer. `RESULT: PASS` means the GPU ensemble
agrees with the CPU reference to ~`1e-12` (double-precision velocity Verlet, same
core on both sides — see THEORY.md §6).

> This is a **reduced-scope teaching model** (CLAUDE.md §13): the quantum region
> is a 2×2 model Hamiltonian, not a DFT solve. It is a software demonstration of
> the QM/MM **force-evaluation + Verlet loop** and of **ensemble GPU
> integration** — not a chemical prediction. All data is synthetic.
