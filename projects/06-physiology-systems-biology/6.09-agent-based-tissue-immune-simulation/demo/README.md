# Demo — 6.9 Agent-Based Tissue / Immune Simulation

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/tissue_params.txt` scenario.
3. **Verify** the GPU result against the CPU reference (`reference_cpu.cpp`):
   the chemokine field total matches **exactly** (integer fixed-point quanta) and
   the final cell positions match within a tiny tolerance.
4. **Time** the run (CUDA events vs a host clock) — a *teaching artifact*, not a
   benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric error (which vary run to run), so
  it is shown but never diffed.

## Reading the output

- `chemokine: total=... peak=... at (col,row)` — the diffused chemokine field.
  The **peak sits over the tumor** at the domain centre (col 15, row 16 for the
  32×32 sample), because the tumor cells secrete there and the field diffuses out.
- `chemokine total quanta (exact integer)` — the field mass in fixed-point quanta.
  This integer is what the CPU and GPU agree on **exactly** (order-free atomics).
- `mean immune->tumor distance: start=... end=...` — the **science check**. Immune
  cells start scattered (`start≈11.44`) and chemotax inward, so the distance
  **shrinks** (`end≈8.03`). If chemotaxis were off, this would barely change.

## Expected result

```
6.9 -- Agent-Based Tissue / Immune Simulation
grid 32x32 (dx=1.00), 70 cells (40 tumor + 30 immune), 300 steps
chemokine: total=901.229917  peak=7.404112 at (col=15,row=16)
chemokine total quanta (exact integer): 901229917
mean immune->tumor distance: start=11.436174  end=8.025364
RESULT: PASS (GPU field total == CPU exactly; positions within tol=1.0e-06)
```

The stderr timing line will differ on your machine (and the GPU is *slower* here
— the sample is tiny and every step launches several small kernels plus a host
bin rebuild; the GPU's advantage grows with cell count and grid size).
