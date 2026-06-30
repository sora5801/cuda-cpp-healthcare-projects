# Demo — 2.30 Protein Solubility & Phase Separation Simulation

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/system.txt` — 6 synthetic IDP chains
   (36 sticky coarse-grained beads) in a periodic box, integrated for 120
   velocity-Verlet steps on both the CPU and the GPU.
3. **Verify** that the GPU and CPU final-state summaries agree: the same shared
   HPS physics (`hps_model.h`) run in the same fixed pair order, so they match to
   ~1e-15 (machine precision) — well inside the documented tolerance. Integer
   order parameters (`condensed beads`, `max local density`) must match exactly.
4. **Time** the GPU kernels (CUDA events) and the CPU baseline — a *teaching
   artifact*, not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic (fixed-precision printing) and is
  diffed against [`expected_output.txt`](expected_output.txt).
- **stderr** carries timing and the GPU-vs-CPU numeric differences (which vary
  run to run), so it is shown but never diffed.

## Expected result

```
2.30 -- Protein Solubility & Phase Separation Simulation
HPS coarse-grained LLPS model (synthetic, reduced units)
beads=36  chains=6  chain_len=6  box=7.000  steps=120
final potential energy = -11.058823
final kinetic   energy = 1.284242
position checksum      = 346.802401
phase order parameters:
  max  local density (neighbours within r_cut) = 15
  mean local density                           = 10.8333
  condensed beads (>=4 neighbours)             = 36 of 36
RESULT: PASS (GPU matches CPU within tolerance)
```

## How to read it

- **Negative potential energy** (`-11.06`) means the system has fallen into an
  attractive, bound state — the chains stuck together rather than flying apart.
- **All 36 beads condensed** with a high `mean local density` (~10.8 neighbours
  within the cutoff) is the demo's headline: the six chains, started as separated
  rods, coalesced into a **single dense droplet** — the minimal signature of
  liquid-liquid phase separation (LLPS).
- **`RESULT: PASS`** means the GPU trajectory reproduced the CPU trajectory; the
  stderr line shows the actual differences (`|dPE|`, `|dKE|`, `|dchecksum|`).

Try `python scripts/make_synthetic.py --lam 0.2` then rerun: with weak stickiness
the chains stay dispersed (low local density, near-zero condensed beads) — the
**soluble** side of the phase boundary. (That changes the numbers, so regenerate
`expected_output.txt` if you want the demo to keep passing on the new input.)

> This is a **simplified, synthetic** teaching model (reduced units, NVE, all-
> pairs O(N^2) forces, no thermostat, uniform synthetic stickiness) — not a
> condensate prediction engine. See `THEORY.md` for what production HPS/CALVADOS
> codes do differently.
