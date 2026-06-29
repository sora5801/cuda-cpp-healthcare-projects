# Data — 1.23 QM/MM Molecular Dynamics

## Committed sample (`sample/ensemble_params.txt`)

| Field | Value |
|---|---|
| Origin | **Synthetic** ensemble configuration (`scripts/make_synthetic.py`) |
| License | Public domain (CC0) — it is synthetic |
| Size | < 1 KB |
| Contents | One line: integration settings + a 2-D `field × x0` parameter sweep |

The "data" is the **ensemble setup**, not measured input. Each trajectory's
parameters are derived from the sweep grid, so the whole ensemble is fully
reproducible from this single line. The model potential-energy surface itself is
built analytically in [`src/qmmm.h`](../src/qmmm.h) — there is no external file.

> **Everything here is synthetic.** The two-state double-well surface is a
> didactic *model*, **not** a quantum-chemistry potential fitted to a real
> molecule. Nothing in this project is suitable for any chemical or clinical
> claim (CLAUDE.md §1, §8).

### File format (one line)

```
dt  steps  v0  nf  nx  field_lo  field_hi  x0_lo  x0_hi
```

| Field | Meaning | Units |
|---|---|---|
| `dt` | velocity-Verlet timestep | model time |
| `steps` | number of Verlet steps per trajectory | — |
| `v0` | initial proton velocity (shared by all members) | length/time |
| `nf` | number of MM-field values in the sweep | — |
| `nx` | number of initial-position values in the sweep | — |
| `field_lo..hi` | MM electrostatic-embedding field range; more negative tilts the surface toward the product (transfer) | energy/length |
| `x0_lo..hi` | initial proton position range (both inside the donor well) | length |

The ensemble has `nf · nx` trajectories. Committed sample:
`0.004 5000 0 16 16 0 -24 -0.7 -0.5` → **256 trajectories**, 5000 steps each.
At `field = 0` the proton stays trapped in the donor well; as the field grows
more negative the surface tilts until the proton crosses the barrier and
**transfers** to the acceptor well — the qualitative behaviour of an
environment-driven proton transfer.

## "Full dataset" / real QM/MM

Real QM/MM does not read a parameter sweep — it reads molecular structures and
computes the quantum surface on the fly. To go further:

- **Enzyme–drug complexes (PDB):** <https://www.rcsb.org> — crystal structures
  to carve a QM region (substrate + catalytic residues) out of.
- **BRENDA:** <https://www.brenda-enzymes.org> — enzyme reaction/kinetics data.
- **SAMPL challenges:** <https://github.com/samplchallenges> — blind benchmarks
  (binding, reactivity) used to validate QM/MM methods.

Production GPU QM/MM engines (the real version of this project):

- **AMBER + QUICK** (<https://github.com/merzlab/QUICK>) — GPU DFT engine for QM/MM.
- **TeraChem** (<https://www.petachem.com>) — GPU DFT server.
- **OpenMM + PySCF** (<https://github.com/openmm/openmm>) — Python QM/MM interface.
- **CP2K** (<https://github.com/cp2k/cp2k>) — GPU-accelerated periodic QM/MM.

`scripts/download_data.{ps1,sh}` print these pointers; there is nothing to fetch
for the demo. Bigger synthetic ensemble:
`python scripts/make_synthetic.py --nf 64 --nx 64` (4096 trajectories).

## Provenance & honesty

The configuration is **synthetic**; the surface is a model, and the units are a
self-consistent "model" system (not real atomic units — see THEORY.md). Outputs
are a software demonstration of the **QM/MM force-evaluation + Verlet loop** and
of **ensemble GPU integration**, not a chemical prediction.
