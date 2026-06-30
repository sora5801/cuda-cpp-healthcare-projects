# Data — 2.19 Membrane Protein Simulation

> **Everything here is SYNTHETIC.** No patient data, no real structures, no
> clinical content. The simulation builds its own coarse-grained membrane patch
> from a handful of parameters; this folder only holds that parameter file.

## Committed sample (`sample/membrane_sample.txt`)

| Field | Value |
|---|---|
| File | `sample/membrane_sample.txt` |
| Origin | **Synthetic** — written by `scripts/make_synthetic.py` |
| License | Public domain (CC0) — it is synthetic |
| Size | < 1 KB |

This tiny file lets `demo/run_demo` run **offline, with zero downloads** (a hard
requirement, CLAUDE.md §8). The geometry of the bilayer + protein is **not** in
this file — it is generated deterministically in code (`build_system()` in
`src/reference_cpu.cpp`). The file holds only the run **parameters**.

### File format

Lines beginning with `#` are comments (ignored by the loader). The two numeric
records are:

```
n_lipids n_prot box_x box_y sigma rcut k_bond r_bond dt steps temp gamma seed
eHH eHT eHP eTT eTP ePP
```

| Symbol | Meaning | Units (reduced MD) |
|---|---|---|
| `n_lipids` | number of lipids (each = 1 HEAD + 2 TAIL beads) | count |
| `n_prot` | number of protein beads (a transmembrane column) | count |
| `box_x`, `box_y` | periodic box size in the membrane plane | σ |
| `sigma` | Lennard-Jones bead diameter | σ (= 1) |
| `rcut` | LJ cutoff radius (pairs beyond this are ignored) | σ |
| `k_bond` | harmonic bond stiffness (lipid springs) | ε/σ² |
| `r_bond` | bond rest length | σ |
| `dt` | integration timestep | τ |
| `steps` | number of MD steps | count |
| `temp` | thermostat target temperature `kT` | ε |
| `gamma` | Langevin friction coefficient | 1/τ |
| `seed` | master RNG seed (deterministic random forces) | integer |
| `eHH … ePP` | the 6 unique LJ well depths of the symmetric 3×3 type matrix | ε |

The bead types are **HEAD** (polar lipid head), **TAIL** (hydrophobic lipid
tail), **PROT** (protein). The well-depth matrix encodes the hydrophobic effect:
**TAIL–TAIL** (`eTT`) is the strongest attraction, which is what drives the two
leaflets to stack into a bilayer.

> "Reduced units" mean lengths are measured in σ, energies in ε, time in
> τ = √(mσ²/ε). A rough mapping to real CG-MARTINI numbers (σ ≈ 0.47 nm,
> ε ≈ 2–5 kJ/mol, τ ≈ a few ps) is in `THEORY.md`.

### Regenerate / resize

```
python scripts/make_synthetic.py                                  # the default sample
python scripts/make_synthetic.py --n-lipids 32 --n-prot 7 --steps 400
```

## Full / real-world datasets (not required, not downloaded)

A *real* membrane-protein simulation starts from an experimental structure and
an automated system builder; that is far beyond this teaching model, so we do
**not** fetch any of these. They are listed for the curious learner
(`scripts/download_data.*` prints the same links):

- **MemProtMD** — 3133 membrane proteins inserted in lipid bilayers —
  <https://memprotmd.bioch.ox.ac.uk>
- **GPCRdb** — GPCR structures and MD data — <https://gpcrdb.org>
- **OPM** — Orientations of Proteins in Membranes — <https://opm.phar.umich.edu>
- **CGMD Platform** benchmark systems —
  <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7765266/>

To turn one of those structures into a runnable system you would use
**CHARMM-GUI Membrane Builder** (<https://charmm-gui.org>) or **packmol-memgen**
to place lipids, solvate, and assign a CHARMM36/MARTINI force field. Respect each
resource's license; none of their data is redistributed here.
