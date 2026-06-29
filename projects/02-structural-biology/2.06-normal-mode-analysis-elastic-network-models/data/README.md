# Data — 2.06 Normal Mode Analysis / Elastic Network Models

## Committed sample (`sample/protein_ca.txt`)

| Field | Value |
|---|---|
| Origin | **Synthetic** Cα structure (`scripts/make_synthetic.py`, seed 2) |
| License | Public domain (CC0) — synthetic |
| Contents | 60 Cα atoms (a compact synthetic fold), ANM cutoff 13 Å |

### File format

```
<N> <cutoff>        # number of Cα atoms; ANM spring cutoff (Angstrom)
<x y z>             # one Cα coordinate per line (Angstrom)
... (N lines)
```

The structure is a compact synthetic backbone (Cα-Cα spacing ~3.8 Å, folded into a
blob) so the elastic network is well-connected — yielding exactly the 6 rigid-body
(zero) modes and clean low-frequency functional modes.

## Full dataset

Real NMA uses Cα coordinates from experimental structures:

- **RCSB PDB** (<https://www.rcsb.org>) — download a `.pdb`/`.cif`, extract the
  `CA` atoms' x/y/z, prepend `N cutoff`.
- **ProDy** (<https://github.com/prody/ProDy>) — Python ANM/GNM (parses PDB for you).
- **AlphaFold DB** (<https://alphafold.ebi.ac.uk>) — predicted structures.

Bigger structure: `python scripts/make_synthetic.py --N 120`. (Cutoff 12–15 Å is
typical for Cα ANM.)

## Provenance & honesty

The structure is **synthetic** (a random compact fold), not a real protein, and the
mode/mobility numbers carry no biological meaning. It exists to make the NMA result
interpretable (6 zero modes, flexible loops most mobile) and the cuSOLVER-vs-CPU
comparison verifiable.
