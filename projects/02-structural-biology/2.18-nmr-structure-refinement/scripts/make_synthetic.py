#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic NMR-restraint sample
# ---------------------------------------------------------------------------
# Project 2.18 : NMR Structure Refinement
#
# WHY THIS EXISTS
#   Real NOE restraint lists come from assigned NMR spectra (BMRB / PDB; see
#   data/README.md). Those are large and licensed per-entry, so for an offline,
#   reproducible demo we synthesise a tiny restraint set from a KNOWN target
#   structure. This makes the demo INTERPRETABLE (PATTERNS.md section 6): because
#   we built the restraints from a real 3-D fold, we know a satisfying structure
#   exists, so simulated annealing should drive replicas to a low-energy,
#   all-restraints-satisfied conformation -- and the demo can report that.
#
#   The data is SYNTHETIC and labelled synthetic everywhere. We never claim the
#   recovered fold is a real protein structure.
#
# THE TARGET FOLD
#   A short alpha-helix of `n_beads` Calpha atoms. A canonical alpha-helix has
#   ~3.6 residues per turn, a rise of ~1.5 A per residue, and a Calpha-Calpha
#   virtual-bond length of ~3.8 A. We place beads on that helix, then emit NOE
#   upper-bound restraints for every non-bonded pair whose true Calpha-Calpha
#   distance is below an NOE cutoff (~6 A) -- exactly the short-range contacts a
#   real NOESY experiment reports. The bond restraints (consecutive beads at
#   ~3.8 A) are added by the program itself, not listed here.
#
# OUTPUT FORMAT (matches load_config in src/reference_cpu.cpp)
#   line 1:  n_beads n_restraints bond_len k_bond k_noe
#   line 2:  n_replicas n_steps T_hot T_cold step_sigma base_seed
#   next n_restraints lines:  i j upper
#
# USAGE
#   python scripts/make_synthetic.py                 # default tiny helix
#   python scripts/make_synthetic.py --n-beads 16 --replicas 256
# ===========================================================================
import argparse
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "restraints.txt"

# Canonical alpha-helix geometry (Calpha trace), in Angstrom / radians.
HELIX_RADIUS   = 2.3      # Calpha helix radius
HELIX_RISE     = 1.5      # rise per residue along the helix axis
RES_PER_TURN   = 3.6      # residues per full turn
BOND_LEN       = 3.8      # ideal Calpha(i)-Calpha(i+1) virtual-bond length
NOE_CUTOFF     = 6.0      # emit a restraint when the true distance is below this
NOE_PADDING    = 0.5      # set the upper bound a little above the true distance


def helix_coords(n_beads):
    """Return the (x,y,z) of n_beads Calpha atoms on a canonical alpha-helix."""
    dtheta = 2.0 * math.pi / RES_PER_TURN     # angle advanced per residue
    pts = []
    for i in range(n_beads):
        theta = i * dtheta
        pts.append((HELIX_RADIUS * math.cos(theta),
                    HELIX_RADIUS * math.sin(theta),
                    i * HELIX_RISE))
    return pts


def dist(a, b):
    return math.sqrt(sum((a[k] - b[k]) ** 2 for k in range(3)))


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic NMR restraint sample.")
    ap.add_argument("--n-beads", type=int, default=12, help="Calpha atoms in the chain")
    ap.add_argument("--replicas", type=int, default=512, help="independent SA trajectories")
    ap.add_argument("--steps", type=int, default=4000, help="Monte-Carlo steps per replica")
    ap.add_argument("--t-hot", type=float, default=5.0, help="starting annealing temperature")
    ap.add_argument("--t-cold", type=float, default=0.02, help="final annealing temperature")
    ap.add_argument("--sigma", type=float, default=1.2, help="trial-move std-dev (A)")
    ap.add_argument("--k-bond", type=float, default=10.0, help="bond force constant")
    ap.add_argument("--k-noe", type=float, default=5.0, help="NOE force constant")
    ap.add_argument("--seed", type=int, default=20260629, help="global RNG base seed")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    nb = args.n_beads
    pts = helix_coords(nb)

    # Emit one NOE upper-bound restraint per non-bonded pair within the cutoff.
    # Bonded neighbours (j == i+1) are handled by the bond term, not as NOEs.
    restraints = []
    for i in range(nb):
        for j in range(i + 2, nb):                       # skip i, i+1 (bonded)
            d = dist(pts[i], pts[j])
            if d <= NOE_CUTOFF:
                upper = round(d + NOE_PADDING, 3)        # NOE = upper bound only
                restraints.append((i, j, upper))

    lines = []
    lines.append(f"{nb} {len(restraints)} {BOND_LEN:g} {args.k_bond:g} {args.k_noe:g}")
    lines.append(f"{args.replicas} {args.steps} {args.t_hot:g} {args.t_cold:g} "
                 f"{args.sigma:g} {args.seed}")
    for (i, j, upper) in restraints:
        lines.append(f"{i} {j} {upper:g}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}")
    print(f"[make_synthetic]   chain={nb} beads, {len(restraints)} NOE restraints "
          f"(from an alpha-helix target), {args.replicas} replicas x {args.steps} steps")
    print(f"[make_synthetic]   SYNTHETIC data -- not a real protein structure.")


if __name__ == "__main__":
    main()
