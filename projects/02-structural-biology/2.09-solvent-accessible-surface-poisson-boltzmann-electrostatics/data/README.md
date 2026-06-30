# Data — 2.9 Solvent-Accessible Surface & Poisson-Boltzmann Electrostatics

## Committed sample (`sample/molecule.pqr`)

| Field | Value |
|---|---|
| File | `sample/molecule.pqr` |
| Origin | **Synthetic** dipolar "molecule" (`scripts/make_synthetic.py`) |
| License | Public domain (CC0) — it is synthetic |
| Size | < 1 KB (11 atoms) |
| Setup | 48³ grid, h=0.6 Å, ε_in=2, ε_out=80, κ²=0.1, 600 sweeps |

This tiny file lets `demo/run_demo` run **offline, with zero downloads**
(CLAUDE.md §8). It is `.pqr`-style: each atom carries a **partial charge (q)** and
a **radius** — the exact information continuum-electrostatics solvers need, and
what **PDB2PQR** produces from a real PDB structure.

The "data" is really the **molecular input**; the dielectric / screening / charge
grids are then *built* from the atoms by `build_problem` in
`src/reference_cpu.cpp`.

### File format

```
line 1:  natoms  n  h  eps_in  eps_out  kappa2  iters
then natoms lines:  x  y  z  q  radius
```

| Field | Meaning | Units |
|---|---|---|
| `natoms` | number of atom rows that follow | — |
| `n` | grid cells per side (the grid is `n³`) | — |
| `h` | grid spacing (distance between neighbours) | Å |
| `eps_in` | dielectric constant inside the protein (~2–4) | — |
| `eps_out` | dielectric constant of water (~80) | — |
| `kappa2` | squared inverse Debye length (∝ ionic strength); 0 = pure Poisson | 1/Å² |
| `iters` | red-black Gauss–Seidel sweeps | — |
| `x y z` | atom position | Å |
| `q` | partial charge (may be negative) | e (elementary charge) |
| `radius` | atomic / van-der-Waals radius (defines low-ε interior) | Å |

The committed molecule is two charged lobes (net +1 e at x≈−4, −1 e at x≈+4)
joined by a neutral scaffold → a clean **dipole** whose solved potential is
antisymmetric (see the demo's `min = −max`, centre ≈ 0).

## Full dataset — using a real protein

There is **nothing to download** for the demo. To run on a real structure:

- **Source:** RCSB PDB — <https://www.rcsb.org> (open, freely redistributable
  coordinates).
- **Prepare charges + radii:** PDB2PQR — <https://github.com/Electrostatics/pdb2pqr>
  (`pdb2pqr30 --ff=AMBER in.pdb out.pqr`).
- **Reformat:** map the resulting `.pqr` ATOM columns (`x y z q radius`) into this
  project's one-line-header format above.
- **Validation benchmark:** APBS test set —
  <https://github.com/Electrostatics/apbs> (compare your field to a reference PB
  solver). Catalog also lists pKDBD (protein pKa values; verify URL) and
  BindingMOAD — <https://bindingmoad.org> — for electrostatic benchmarking.

`scripts/download_data.ps1` / `.sh` print these instructions and never bypass any
credentials. For a bigger synthetic problem with no download:
`python scripts/make_synthetic.py --n 64 --iters 800`.

## Provenance & honesty

The sample is **synthetic** and abstract — an idealized dipole, not a real
molecule. Its potentials are in **reduced (illustrative) units** and demonstrate
the finite-difference / red-black-stencil method only. They are **not** calibrated
pKa values, binding energies, or zeta potentials, and imply **no** clinical or
quantitative validity. Synthetic data is labelled synthetic everywhere it appears.
