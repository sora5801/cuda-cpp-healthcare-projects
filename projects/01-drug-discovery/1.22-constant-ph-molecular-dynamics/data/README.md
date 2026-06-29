# Data — 1.22 Constant-pH Molecular Dynamics

## Committed sample (`sample/cph_system.txt`)

| Field | Value |
|---|---|
| File | `sample/cph_system.txt` |
| Origin | **SYNTHETIC** — hand-authored 3-residue toy, **not a real protein** |
| License | Public domain (CC0) — it is synthetic, invented for teaching |
| Size | < 1 KB |
| Regenerate | `python scripts/make_synthetic.py` (writes this exact file) |

This tiny file lets `demo/run_demo` run **offline, with zero downloads** — a hard
requirement for every project (CLAUDE.md §8). It is **synthetic**: the residues,
positions, charges, and intrinsic pKa values are stylised teaching values, not
measurements of any real molecule. **No clinical or experimental meaning.**

### Why this particular toy

The system is engineered so the result is *interpretable*: an acid (ASP) flanked
on a line by two bases (HIS, LYS). Because the electrostatic coupling is switched
**on**, each residue's apparent pKa visibly **shifts** away from its intrinsic
value — which is exactly the effect constant-pH simulation exists to capture:

- **ASP** (acid) sits next to two cations → its negative (deprotonated) form is
  stabilised → its pKa shifts **down** (~−1.9 in the demo).
- **LYS** (base) sits next to ASP's negative charge → its positive (protonated)
  form is stabilised → its pKa shifts **up** (~+0.7).
- **HIS** (in the middle) feels an acid on one side and a base on the other; the
  shifts roughly cancel (~0).

### File format (consumed by `src/reference_cpu.cpp` `load_cph_problem`)

`#` starts a comment to end of line. Tokens are whitespace/newline separated.

```
line 1:  n_res  coulomb_k  kT       sweeps  burn_in
line 2:  pH_min pH_max     n_pH     replicas seed
then n_res lines:  pKa_intrinsic  q_prot  q_deprot   x   y   z
```

| Field | Meaning | Units |
|---|---|---|
| `n_res` | number of titratable residues | count (≤ 16) |
| `coulomb_k` | Coulomb prefactor `332.06/epsilon`; **set 0 to disable coupling** | kcal·Å·mol⁻¹·e⁻² |
| `kT` | thermal energy `k_B·T` (≈ 0.593 at 298 K) | kcal/mol |
| `sweeps` | Monte Carlo sweeps per chain (1 sweep = `n_res` attempted flips) | count |
| `burn_in` | leading sweeps discarded before tallying (equilibration) | count |
| `pH_min`,`pH_max`,`n_pH` | the pH grid the titration curve is sampled on | pH units / count |
| `replicas` | independent chains averaged per pH (Monte Carlo noise reduction) | count |
| `seed` | base RNG seed (chain `(k,r)` seeds from `(seed, chain_id(k,r))`) | integer |
| `pKa_intrinsic` | the residue's model pKa in isolation (Asp≈4, His≈6.5, Lys≈10.5) | dimensionless |
| `q_prot` / `q_deprot` | net charge of the protonated / deprotonated form | units of `e` |
| `x`,`y`,`z` | fixed residue position (frozen — see THEORY.md) | Ångström |

## Full / real datasets (pointers only — not redistributed here)

The catalog points at real titration benchmarks. We do **not** commit them: they
are tied to specific PDB structures and curated reference values whose
redistribution terms vary. Study them to extend this project (see Exercises):

- **PKAD** — experimental pKa database for protein residues:
  <https://compbio.clemson.edu/pkad/>
- **PHMD / benchmark pKa sets** for Asp/Glu/His/Cys/Lys residues (used to validate
  CpHMD methods in the literature).
- **DrugBank** — compounds with ionizable groups (for ligand pKa work):
  <https://go.drugbank.com> (registration required).

`scripts/download_data.ps1` / `.sh` print these links and instructions; they do
**not** attempt to bypass any registration or license (CLAUDE.md §8). If you map a
real residue into this toy, replace `pKa_intrinsic`, charges, and positions
accordingly and **keep labeling derived inputs honestly**.

## Honesty

Everything in `sample/` is synthetic and labeled synthetic. Nothing here is a
measurement, and no output may be used for any real chemical or medical decision.
