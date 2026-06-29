#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate a synthetic protein-ligand batch
# ---------------------------------------------------------------------------
# Project 1.15 : Protein-Ligand Binding Affinity Scoring (ML)
#
# WHY SYNTHETIC
#   The real training/benchmark data for ML scoring functions are PDBbind
#   (~19,000 protein-ligand complexes with measured Kd/Ki) and CASF-2016 -- both
#   require registration and have redistribution terms (see data/README.md and
#   scripts/download_data.*). To keep the demo OFFLINE, reproducible, and free of
#   any licensing question, we generate a clearly-SYNTHETIC batch of "docked
#   poses": a small protein scaffold plus a ligand of a few atoms, all inside the
#   16 A grid box. A fixed RNG seed makes the output byte-for-byte reproducible so
#   demo/expected_output.txt is stable.
#
#   This data is chemically meaningless. It exists ONLY to exercise and VERIFY the
#   3D-CNN forward pass (CPU vs GPU). The `label` we attach is a toy "true pKd"
#   correlated with ligand size; the model does NOT see it, and no chemical or
#   clinical conclusion may be drawn (CLAUDE.md sec.8).
#
# OUTPUT FORMAT (data/README.md):
#   line 1            : "<n>"                                  number of complexes
#   then, per complex : "<m> <label_pKd>"                      m = atom count
#                       m lines of "<x> <y> <z> <type> <is_ligand>"
#   type in {0=C,1=N,2=O,3=S}; is_ligand in {0,1}; coords in angstroms in [0,16).
#
# USAGE
#   python scripts/make_synthetic.py                 # default n=6 complexes
#   python scripts/make_synthetic.py --n 100000      # a "rescore at scale" batch
# ===========================================================================
import argparse
import math
import random
from pathlib import Path

# These MUST match constants in src/scoring_core.h (the loader rejects out-of-box
# atoms only implicitly -- voxels outside [0,GRID) just get no density -- but we
# keep generation inside the box so every atom contributes).
BOX_A = 16.0          # grid spans 16 angstroms per side
NTYPES = 4            # C, N, O, S

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "complexes_sample.txt"


def rand_atom(rng, cx, cy, cz, spread, is_ligand):
    """One atom near center (cx,cy,cz) with Gaussian scatter `spread` (A),
    a random element type, clamped into the [0.5, BOX_A-0.5) box so it stays on
    the grid. Returns (x, y, z, type, is_ligand)."""
    def clamp(v):
        return min(BOX_A - 0.5, max(0.5, v))
    x = clamp(cx + rng.gauss(0.0, spread))
    y = clamp(cy + rng.gauss(0.0, spread))
    z = clamp(cz + rng.gauss(0.0, spread))
    t = rng.randrange(NTYPES)
    return (x, y, z, t, is_ligand)


def make_complex(rng, n_protein, n_ligand):
    """Build one synthetic complex: a protein scaffold cloud (is_ligand=0) plus a
    compact ligand cloud (is_ligand=1) sharing the binding-pocket center. Returns
    (atoms, label) where label is a toy pKd correlated with ligand size."""
    # Pocket center somewhere in the middle third of the box.
    cx = rng.uniform(BOX_A / 3, 2 * BOX_A / 3)
    cy = rng.uniform(BOX_A / 3, 2 * BOX_A / 3)
    cz = rng.uniform(BOX_A / 3, 2 * BOX_A / 3)
    atoms = []
    # Protein scaffold: broad cloud around the pocket.
    for _ in range(n_protein):
        atoms.append(rand_atom(rng, cx, cy, cz, spread=3.0, is_ligand=0))
    # Ligand: tight cloud in the pocket.
    for _ in range(n_ligand):
        atoms.append(rand_atom(rng, cx, cy, cz, spread=1.2, is_ligand=1))
    # Toy "true" pKd: larger ligands buried in the pocket bind tighter, plus a
    # little noise. Range ~[4, 10]. NOT used by the model -- interpretability only.
    label = 4.0 + 0.25 * n_ligand + rng.gauss(0.0, 0.3)
    label = max(2.0, min(11.0, label))
    return atoms, label


def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic protein-ligand batch.")
    ap.add_argument("--n", type=int, default=6, help="number of complexes (poses)")
    ap.add_argument("--seed", type=int, default=11, help="RNG seed (determinism)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    lines = [str(args.n)]
    for i in range(args.n):
        # Vary the ligand size 4..(4+n-1) so the predicted-affinity table spans a
        # range and the rank-1 binder is a meaningful, deterministic outcome.
        n_ligand = 4 + (i % 8)
        n_protein = 24 + rng.randrange(8)
        atoms, label = make_complex(rng, n_protein, n_ligand)
        lines.append(f"{len(atoms)} {label:.3f}")
        for (x, y, z, t, lig) in atoms:
            lines.append(f"{x:.4f} {y:.4f} {z:.4f} {t} {lig}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  (n={args.n} complexes; SYNTHETIC, seed={args.seed})")


if __name__ == "__main__":
    main()
