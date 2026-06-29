# Data — 1.8 Semi-Empirical & Tight-Binding Quantum Methods

## Committed sample (`sample/`)

| Field | Value |
|---|---|
| File | `sample/molecules_sample.txt` |
| Origin | **Synthetic** — idealised π-system connectivity graphs of textbook molecules |
| License | Public domain (CC0) — it is synthetic; the molecules are common knowledge |
| Size | < 1 KB |
| Generator | `python scripts/make_synthetic.py` (deterministic) |

This tiny file lets `demo/run_demo` run **offline, with zero downloads**, a hard requirement for every
project (CLAUDE.md §8). It is **SYNTHETIC**: there are no 3-D coordinates and no experimental data — only
each molecule's π-bond **connectivity graph**, which is the only input a Hückel/tight-binding model needs.
The molecules (ethylene, allyl, butadiene, benzene, cyclobutadiene, hexatriene, cyclopentadienyl,
naphthalene) were picked because their Hückel spectra are **known in closed form**, so the demo can check
itself against analytic chemistry, not just CPU-vs-GPU agreement.

### File format

```
# lines beginning with '#' are comments
NUM_MOL                       # number of molecules in the batch
# then, repeated NUM_MOL times:
NAME  N  NBONDS               # molecule name, atom count, bond count
i j                           # NBONDS lines, each a 0-based bond between π-atoms i and j
...
```

**Field meanings.** `NAME` is a human label (also printed in the report). `N` is the number of sp² carbons
= number of π orbitals = basis size. Each `i j` line is an **undirected π bond** (stored symmetrically).
There are **no units** — the graph alone defines the Hückel Hamiltonian (`H_ii = α`, `H_ij = β` if bonded;
the code uses the textbook scale `α = 0`, `β = −1`, so energies are reported in units of `|β|`).

## Full dataset (real, not committed)

The real semi-empirical / tight-binding workflow is run on large molecular datasets that ship 3-D
geometries (and often reference DFT energies). They are **large and require downloading**; we do **not**
redistribute them. `scripts/download_data.ps1` / `.sh` print fetch instructions and never bypass any
registration or license.

- **ANI-1** — ~20M DFT energy calculations on 57k small organic molecules.
  <https://github.com/isayev/ANI1> (CC0 / see repo). Used to benchmark fast methods against DFT.
- **QM9** — 134k small molecules with DFT properties.
  <https://doi.org/10.6084/m9.figshare.978904> (CC0).
- **GMTKN55** — benchmark thermochemistry / kinetics / non-covalent sets.
  <https://www.chemie.uni-bonn.de/grimme/de/software/gmtkn> (academic use; see site).
- **COMPAS** — computational database of polycyclic aromatic systems (verify current URL).

To use any of these here you would convert each molecule's geometry to a π-system graph (a chemistry
toolkit such as RDKit/Open Babel does the perception); that conversion is out of scope for this teaching
project, which takes the graph directly.

## Honesty & safety

- The committed sample is **synthetic** and labelled synthetic everywhere it appears.
- Energies are a **relative `|β|` scale**, not kcal/mol — no physical-units claim is made.
- Nothing here is clinically valid; this is study material only (CLAUDE.md §8).
