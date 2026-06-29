# Demo — 14.02 Spatial Reaction-Diffusion (Gray-Scott)

## What this demonstrates

`run_demo.ps1` (Windows) / `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on `data/sample/grayscott_params.txt` (128×128 grid, 8000 steps).
3. **Verify** that the GPU stencil produces the **same final field** as the CPU
   reference (within a tight floating-point tolerance).
4. **Report** pattern metrics (total V, peak V, active-cell coverage) and a
   profile along the centre row.

stdout (the metrics) is deterministic and diffed against
[`expected_output.txt`](expected_output.txt); the timing line is on stderr only.

## Canonical output

See [`expected_output.txt`](expected_output.txt). From a tiny central seed, the V
field self-organizes into a **Turing labyrinth** covering ~half the grid (total V
≈ 3345, ~8600 active cells). `RESULT: PASS` means the GPU and CPU fields agree to
~`1e-7`. The GPU runs the 8000-step simulation ~16× faster than the CPU; the edge
grows with grid size.

> Gray-Scott is an **abstract** two-chemical model (and this is the continuum
> grid version of the molecular-resolution catalog project) — a demonstration of
> the reaction-diffusion / stencil pattern, not a biochemical simulation.
