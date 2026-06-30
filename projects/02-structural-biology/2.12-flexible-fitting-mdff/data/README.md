# Data — 2.12 Flexible Fitting / MDFF

## Committed sample (`sample/mdff_problem.txt`)

| Field | Value |
|---|---|
| File | `sample/mdff_problem.txt` |
| Origin | **Synthetic** fitting problem (`scripts/make_synthetic.py`) |
| License | Public domain (CC0) — it is synthetic |
| Size | < 2 KB |
| Setup | 27-atom 3×3×3 lattice, misfitted from its target inside a 32³ density map |

The "data" is the **fitting problem**, not a patient map: a small atomic model
that has been displaced from its target, together with the parameters of a
cryo-EM-style density *simulated* from that target. The program rebuilds the
density grid from the targets, so the committed file stays tiny (atoms, not a
full 24³ voxel grid). This lets `demo/run_demo` run **offline, zero downloads**
(CLAUDE.md §8).

### File format

```
nx ny nz vox w_dens k_rest step iters natoms sigma      <- line 1 (header)
x0_x x0_y x0_z                                          <- natoms start lines
...
tx   ty   tz                                            <- natoms target lines
...
```

| Field | Meaning | Units |
|---|---|---|
| `nx ny nz` | density-map grid dimensions | voxels |
| `vox` | voxel size (isotropic); grid origin at world (0,0,0) | world-units/voxel |
| `w_dens` | weight on the density (fitting) force | force per density-slope |
| `k_rest` | harmonic-restraint stiffness (MD-force-field stand-in) | force/length |
| `step` | overdamped steepest-descent step size | length·time/force |
| `iters` | number of fitting iterations | — |
| `natoms` | number of atoms in the model | — |
| `sigma` | Gaussian blob width used to simulate the density | world units |
| `x0_*` | **starting** (misfitted) atom position | world units |
| `t*` | **ground-truth target** position (answer key; scoring only) | world units |

Default sample: `32 32 32 1 6 0.05 0.05 200 27 1.2` → the model starts ~1.41 units
RMSD from target; the fit pulls it onto the density ridges (final RMSD ~0.03).
The lattice is spaced (L=6) with narrow blobs (sigma=1.2) so each atom's density
basin is separate — overlapping basins would make atoms collapse to the centre
(a real MDFF over-fitting failure mode discussed in THEORY).

Regenerate or resize: `python scripts/make_synthetic.py --iters 400`.

## "Full dataset" / real cryo-EM fitting

Real MDFF fits an experimental **cryo-EM density map** (EMDB) and a starting
**atomic model** (PDB) — parsing MRC/CCP4 maps and PDB coordinates:

- **EMDB** — reference density maps: <https://www.ebi.ac.uk/emdb/>
- **EMPIAR** — raw particle data: <https://www.ebi.ac.uk/empiar/>
- **Ribosome MDFF benchmarks** — PDB `3J7Y`, `4V6X`.
- **Viral capsid** fitting datasets (large complexes that motivate the GPU).

`scripts/download_data.ps1` / `.sh` print these pointers; there is nothing to
download for the demo (the sample is self-contained). Wiring a real map would add
an MRC/CCP4 reader and a PDB reader, then feed `rho` and `x0` straight into the
same kernel (see THEORY "Where this sits in the real world").

## Provenance & honesty

The model is a **synthetic lattice**, not a biomolecule, and the density is a
simple Gaussian sum, not an experimental map. It demonstrates the MDFF /
trilinear-gather GPU pattern; it is **not** a validated structure-determination
pipeline and **not for clinical use**. Synthetic data is labelled synthetic
everywhere it appears.
