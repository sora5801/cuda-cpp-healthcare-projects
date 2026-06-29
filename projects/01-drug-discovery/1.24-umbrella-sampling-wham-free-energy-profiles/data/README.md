# Data — 1.24 Umbrella Sampling / WHAM Free Energy Profiles

## Committed sample (`sample/umbrella.txt`)

| Field | Value |
|---|---|
| File | `sample/umbrella.txt` |
| Origin | **Synthetic** experiment configuration (`scripts/make_synthetic.py`) |
| License | Public domain (CC0) — it is synthetic |
| Size | < 1 KB |

The "data" here is the **experiment setup**, not measured input: a synthetic
double-well landscape plus the umbrella-window layout and Langevin-dynamics
settings. The program runs the biased dynamics, histograms the reaction
coordinate per window, and reconstructs the potential of mean force (PMF) with
WHAM. Everything is reproducible from this one file plus the RNG seed, so the
demo runs **offline with zero downloads** (CLAUDE.md §8).

> **Synthetic, educational only.** The double-well `U(x) = A (x²−b²)²/b⁴` is a
> textbook two-state model (one barrier, two minima) — not a real molecule, not a
> real channel, and nothing here is clinical.

### File format (four whitespace-separated lines)

```
A b
x_min x_max nbins
n_windows win_min win_max k_spring
D dt n_equil n_sample seed
```

| Field | Meaning | Units |
|---|---|---|
| `A` | barrier height of the double-well at `x=0` | kT |
| `b` | half-separation of the wells (minima at `±b`) | reaction-coord. |
| `x_min`,`x_max`,`nbins` | histogram grid: range and number of equal bins | reaction-coord., — |
| `n_windows` | number of umbrella windows (independent biased simulations) | — |
| `win_min`,`win_max` | first and last restraint center `x0` (evenly spaced) | reaction-coord. |
| `k_spring` | harmonic spring constant shared by all windows | kT / x² |
| `D` | diffusion constant of the overdamped Langevin dynamics | x² / time |
| `dt` | Langevin timestep | time |
| `n_equil` | discarded warm-up steps per window | — |
| `n_sample` | recorded (histogrammed) steps per window | — |
| `seed` | base RNG seed (reproducibility) | — |

Default sample: `A=4, b=1`; grid `[-1.6,1.6]`/32 bins; 27 windows in `[-1.3,1.3]`
with `k_spring=12`; `D=0.05, dt=0.005`, 4000 equilibration + 60000 sample steps,
seed `20240117`. That is 27 × 60000 = 1,620,000 total samples — enough for WHAM to
recover the 4 kT barrier to within ~0.2 kT over the interior of the scan.

We work in **reduced units with kT = 1**, so all energies/PMFs are in units of kT
(the natural unit for free energy). A real run carries kcal/mol or kJ/mol.

## "Full dataset" / real umbrella sampling

Real umbrella sampling restrains a **collective variable** of an all-atom system
(ligand–pocket distance, ion depth in a channel, a dihedral) and runs full
molecular dynamics inside each window. There is no single file to download; you
generate trajectories with an MD engine and post-process them with WHAM. Starting
points from the catalog:

- **GROMACS umbrella-sampling tutorial** — <https://tutorials.gromacs.org> — the
  canonical worked example (pulling a molecule, building windows, `gmx wham`).
- **SAMPL binding free-energy challenges** — <https://github.com/samplchallenges/SAMPL>
  — community blind tests of binding free energies (a target for these methods).
- **BindingDB** — <https://www.bindingdb.org> — measured binding affinities to
  compare computed PMFs against.
- **Ion-channel permeation benchmark sets** — used to validate PMF protocols for
  pore permeation.

Bigger synthetic experiment (no download):
`python scripts/make_synthetic.py --n-windows 51 --n-sample 200000`.

## Provenance & honesty

The configuration is **synthetic**; the landscape and parameters are illustrative,
not fitted to any molecule. Outputs are a software demonstration of umbrella
sampling + WHAM, not a free-energy prediction for any real system.
