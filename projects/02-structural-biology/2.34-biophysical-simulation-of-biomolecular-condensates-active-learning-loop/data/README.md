# Data — 2.34 Biophysical Simulation of Biomolecular Condensates (Active Learning Loop)

## Committed sample (`sample/condensate_ensemble.txt`)

| Field | Value |
|---|---|
| Origin | **Synthetic** experiment configuration (`scripts/make_synthetic.py`) |
| License | Public domain (CC0) — it is synthetic |
| Size | < 1 KB (one line) |
| Contents | the coarse-grained MD model constants + the stickiness sweep + the target D |

The "data" here is the **experiment setup**, not a measured dataset: it is one
line of numbers that fully specifies one active-learning iteration. The program
derives every candidate sequence's stickiness `lambda` from the sweep grid and
runs a deterministic Brownian-dynamics trajectory, so the whole ensemble is
reproducible from this single file. This lets `demo/run_demo` run **offline,
with zero downloads** (CLAUDE.md §8).

### File format (one whitespace-separated line)

```
n_beads steps dt kT gamma k_bond r0 eq_steps lag seed  n_members lambda_lo lambda_hi k_cohese target_D
```

| Field | Meaning (units are reduced MD units: σ length, τ time, kT energy) |
|---|---|
| `n_beads` | beads per coarse-grained chain (a short IDP); ≤ 16 (kernel local-array cap) |
| `steps` | Brownian-dynamics steps per trajectory |
| `dt` | integration timestep |
| `kT` | thermal energy — sets the thermal-noise amplitude |
| `gamma` | friction coefficient (overdamped drag) |
| `k_bond` | harmonic bond stiffness (chain connectivity) |
| `r0` | bond rest length |
| `eq_steps` | equilibration steps discarded before measuring (startup transient) |
| `lag` | MSD time-lag in steps (≤ 24) — the probe for internal mobility `D` |
| `seed` | global RNG seed (counter-based RNG → reproducible) |
| `n_members` | number of candidate sequences (ensemble size) |
| `lambda_lo`, `lambda_hi` | stickiness range scanned (the CALVADOS-style cohesion knob) |
| `k_cohese` | base cohesive stiffness scale shared by all replicas |
| `target_D` | experimental target diffusion coefficient to match by design |

Committed values: `12 500 0.005 1 1 80 1 150 20 20260628 24 0.5 8 2 0.165` →
24 candidate sequences, `lambda ∈ [0.5, 8]`, aiming at `D = 0.165`.

A bigger sweep: `python scripts/make_synthetic.py --n-members 200`.

## "Full dataset" / toward a real loop

This project is a **reduced-scope teaching version** (see `../THEORY.md`). The
real frontier loop the catalog describes pulls from public resources; none is
auto-downloaded (`scripts/download_data.*` prints links, never bypasses terms):

- **PhaSePro** (<https://phasepro.elte.hu>) — curated phase-separating proteins/regions.
- **DisProt** (<https://disprot.org>) — intrinsically disordered regions.
- **RCSB PDB** (<https://www.rcsb.org>) — structures of FUS / TDP-43 / hnRNPA1 LC domains.
- **CALVADOS 2** (<https://github.com/KULL-Centre/CALVADOS>) — residue-level IDP CG model and its
  per-residue stickiness (lambda) table, which our single scalar `lambda` abstracts.

Experimental LLPS partition-coefficient datasets exist in the literature (verify
URL/license before use).

## Provenance & honesty

The configuration is **synthetic** and the parameter ranges are illustrative,
not fitted to any protein. The simulated diffusion coefficients and radii of
gyration are outputs of a deliberately simplified harmonic toy model (THEORY
"Where this sits in the real world"), **not** quantitative predictions for any
real condensate, and carry **no clinical meaning** whatsoever.
