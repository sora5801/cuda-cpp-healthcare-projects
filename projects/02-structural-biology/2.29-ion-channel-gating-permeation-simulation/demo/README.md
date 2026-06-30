# Demo — 2.29 Ion Channel Gating & Permeation Simulation

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/channel_params.txt` input — a tiny,
   clearly-synthetic description of one reduced 1-D ion-channel pore.
3. **Verify** the GPU result against the CPU reference (`reference_cpu.cpp`): the
   two simulate the *identical* Brownian trajectories (shared RNG + per-step
   physics), so the integer tallies must match **exactly** — it prints `PASS`/`FAIL`.
4. **Time** the GPU kernel (CUDA events) and the CPU baseline — a *teaching
   artifact*, not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing (which varies run to run), so it is shown but
  never diffed.

## How to read the result

```
2.29 -- Ion Channel Gating & Permeation Simulation
pore L=3.00 nm, 12 bins, barrier U=4.00 kT (sigma=0.50 nm), q=+1, V=4.00 kT/e
BD: D=0.400 nm^2/step, dt=0.200, steps=2000, ions=256
permeations: forward=63334  reverse=58579  net=4755
net flux per ion-step = 9.287109e-03 (conductance proxy)
occupancy histogram (ion-steps per z-bin):
  92953 93269 75883 47341 24625 11481 6143 6143 11095 24354 46755 71958
peak occupancy at bin 1 (z ~ 0.38 nm)
RESULT: PASS (GPU tallies match CPU exactly)
```

Two things to notice — both *physics you can see in the numbers*:

- **Net forward current.** With a positive applied voltage `V`, the positive ion
  is driven in the `+z` (forward) direction: `forward (63334) > reverse (58579)`,
  giving a **net positive flux** (`+4755` crossings). That net flux is the
  single-channel current — exactly what a patch-clamp electrode measures. Set
  `--voltage 0` in `make_synthetic.py` and the net flux collapses to ~0 (a
  zero-field control): no driving force, no current.
- **The selectivity-filter bottleneck.** The occupancy histogram is **U-shaped**:
  ions pile up in the baths at both mouths (`~93000` at bins 0–1 and `~72000` at
  bin 11) and are **depleted at the pore centre** (`6143` at bins 6–7). That
  central dip is the `4 kT` potential-of-mean-force barrier — the desolvation /
  selectivity-filter region an ion must climb to cross. The histogram *is* the
  free-energy landscape, read out as a probability density.

> **Not a real channel, not clinical.** These parameters describe a didactic
> reduced model, not a specific Nav/Kv/CFTR channel, and carry no medical meaning.

## Why it is deterministic

Every reported quantity is an **integer** (occupancy counts, forward/reverse
crossings). The GPU accumulates them with `atomicAdd`, and integer adds commute,
so the result is independent of thread-scheduling order and equals the CPU's
bit-for-bit. A floating-point current sum would *not* reproduce — see
`../THEORY.md` "Numerical considerations" and `docs/PATTERNS.md §3`.
