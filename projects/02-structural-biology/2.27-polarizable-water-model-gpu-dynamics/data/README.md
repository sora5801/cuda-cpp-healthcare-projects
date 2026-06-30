# Data — 2.27 Polarizable Water Model GPU Dynamics

## Committed sample (`sample/water_cluster.txt`)

| Field | Value |
|---|---|
| File | `sample/water_cluster.txt` |
| Origin | **Synthetic** — a hand-built teaching cluster, NOT experimental data |
| License | Public domain (CC0) — it is synthetic |
| Size | < 1 KB (7 sites) |
| Generator | `scripts/make_synthetic.py` reproduces it byte-for-byte at the default |

This tiny file lets `demo/run_demo` run **offline, with zero downloads** — a hard
requirement for every project (CLAUDE.md §8). It is also the exact content of the
program's built-in fallback system, so the demo's `expected_output.txt` holds
whether or not the file is passed.

### File format

```
line 1 : N  a_thole  max_iters  tol   Eext_x Eext_y Eext_z
next N : x  y  z   q   alpha
```

Lines after a `#` are comments. **Units:** length = Ångström (Å), charge =
elementary charge (e), polarizability `alpha` = Å³, external field `Eext` = e/Å².

### Per-field meaning

| Symbol | Meaning | Units |
|---|---|---|
| `N` | number of sites | — |
| `a_thole` | Thole exponential damping parameter (≈0.39 in AMOEBA) | dimensionless |
| `max_iters` | cap on Jacobi SCF sweeps | — |
| `tol` | convergence threshold on max dipole change | e·Å |
| `Eext` | uniform external electric field applied to every site | e/Å² |
| `x y z` | Cartesian position of the site | Å |
| `q` | permanent point charge | e |
| `alpha` | isotropic polarizability (`0` ⇒ a pure fixed charge, no dipole) | Å³ |

### What the sample contains (engineered to teach)

- **Site 0** — an *isolated* polarizable probe 50 Å from the cluster, sitting in
  the uniform field `Eext = 0.05 ê_z`. Because the cluster's field at 50 Å is
  negligible, its converged dipole must equal the **analytic** `µ = α·E` =
  `1.444 × 0.05 = 0.0722 e·Å`. The demo prints this as a physics cross-check.
- **Sites 1–3 and 4–6** — two TIP4P-like water molecules ~2.9 Å apart (a
  hydrogen-bond-like separation). Their oxygens are the polarizable carriers; the
  hydrogens are fixed charges (`alpha = 0`). At this distance the two waters
  **mutually polarize**, so the self-consistent SCF loop does real work (it does
  not converge after the first "direct" guess). The charges (O = −0.834 e, H =
  +0.417 e) are TIP3P/TIP4P-style; `alpha = 1.444 Å³` is the experimental
  molecular polarizability of water.

> All values are a **simplified teaching model**, not a production force field.
> No clinical or research claim is implied (CLAUDE.md §1, §8).

## Full / real-world reference data

This project does **not** download anything — the synthetic cluster above is
complete. The real-world data that polarizable water models are validated against
is listed in the catalog and printed by `scripts/download_data.{ps1,sh}`:

- **NIST water thermophysical properties** (density, dielectric constant vs. T,P):
  <https://webbook.nist.gov/chemistry/fluid/>. The headline test of a water model
  is reproducing the **density anomaly** (max at 4 °C) and the **dielectric
  constant** — both demand long, well-converged polarizable MD.
- **TIP4P-2005 / SPC/E reference simulations** — community benchmark trajectories
  and parameters for comparing water models.
- **HBond dynamics NMR benchmarks**, **MD2PDB water trajectory archives** — for
  hydrogen-bond lifetime and reorientation comparisons.

Force-field code and parameters (study, don't copy wholesale): **MBX**
(<https://github.com/paesanilab/MBX>), **OpenMM**
(<https://github.com/openmm/openmm>), **Tinker-HP**
(<https://github.com/TinkerTools/tinker-hp>).

Respect every license. None of these are redistributed here; the committed sample
is synthetic precisely so the repo stays clean and offline-runnable.

## Make a bigger synthetic cluster

```
python scripts/make_synthetic.py --waters 64     # probe + 64 waters
```
The generated file uses the same format and labels itself synthetic in a header
comment.
