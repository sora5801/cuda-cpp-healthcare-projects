#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic sample dataset
# ---------------------------------------------------------------------------
# Project 2.23 : Protein-Ligand Interaction Energy Decomposition
#
# WHY THIS EXISTS
#   The real datasets behind this project (PDBbind structures, KLIFS kinase
#   complexes, AMBER prmtop force-field parameters) either require a parser
#   stack we deliberately do not ship for a teaching demo, or carry licenses we
#   must not redistribute. So we DETERMINISTICALLY generate a small, clearly
#   SYNTHETIC stand-in that exercises the exact same physics and loader format,
#   and that has a KNOWN ANSWER so the demo result is interpretable
#   (PATTERNS.md sec 6 -- "engineer the sample so the result is meaningful").
#
# WHAT WE BUILD  (a toy binding pocket around one ligand)
#   * One small ligand of L atoms carrying a net negative charge cluster, placed
#     near the origin and rigidly jittered across F frames (a fake "trajectory").
#   * M protein residue beads arranged around it. ONE residue ("ARG41") is a
#     positively-charged arginine placed right next to the ligand's negative
#     atom -> a SALT BRIDGE, which the decomposition must flag as the #1
#     electrostatic hot spot. A second residue ("LEU88") is a neutral, fat,
#     close-packed leucine -> a vdW (shape) hot spot. The rest are weak/distant.
#   This embeds the textbook lesson: electrostatic hot spots (salt bridges,
#   H-bonds) vs van der Waals hot spots (hydrophobic packing).
#
#   Output layout (parsed by load_system in reference_cpu.cpp; see data/README.md):
#       F M L cutoff
#       M residue rows:  name charge eps rmin_half born
#       L ligand  rows:  charge eps rmin_half born
#       F frames, each:  M residue 'x y z' lines then L ligand 'x y z' lines
#
#   The generator is seeded, so re-running reproduces byte-identical data -> the
#   committed sample and demo/expected_output.txt stay in lockstep.
#
# USAGE
#   python scripts/make_synthetic.py                 # default tiny sample
#   python scripts/make_synthetic.py --residues 64 --frames 50   # bigger
# ===========================================================================
import argparse
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "complex_sample.txt"


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic protein-ligand sample.")
    ap.add_argument("--residues", type=int, default=12, help="number of protein residue beads (M)")
    ap.add_argument("--ligand-atoms", type=int, default=4, help="number of ligand atoms (L)")
    ap.add_argument("--frames", type=int, default=6, help="number of trajectory frames (F)")
    ap.add_argument("--cutoff", type=float, default=12.0, help="interaction cutoff (Angstrom)")
    ap.add_argument("--seed", type=int, default=20260629, help="RNG seed (determinism)")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    M, L, F = args.residues, args.ligand_atoms, args.frames

    # --- Ligand parameters (constant across frames) ----------------------
    # A 4-atom toy fragment: one strongly negative "carboxylate" oxygen (atom 0)
    # that will salt-bridge to ARG41, plus three mild atoms. eps/rmin_half/born
    # are physically plausible AMBER-ish values (well depth ~0.1-0.2 kcal/mol).
    lig = []
    lig.append(dict(q=-0.80, eps=0.21, rmin=1.66, born=1.50))   # atom 0: carboxylate O-
    for a in range(1, L):
        lig.append(dict(q=-0.10, eps=0.11, rmin=1.91, born=1.70))  # mild carbons

    # --- Residue parameters ----------------------------------------------
    # We hand-place two hot spots and fill the rest with weak, distant residues.
    res = []
    res.append(dict(name="ARG41", q=+0.85, eps=0.17, rmin=1.91, born=1.85))  # salt-bridge +
    res.append(dict(name="LEU88", q=+0.00, eps=0.45, rmin=2.10, born=2.00))  # vdW packer
    for m in range(2, M):
        # Weak, slightly polar background residues with random small charge.
        q = rng.uniform(-0.30, 0.30)
        res.append(dict(name=f"GLY{m:02d}", q=round(q, 3), eps=0.10, rmin=1.85, born=1.70))

    # --- Geometry: base coordinates (frame 0), then jitter per frame -----
    # Ligand atom 0 (the carboxylate O-) sits at the +x tip; the remaining mild
    # atoms TRAIL OFF in the -x direction. This matters: it leaves the +x side of
    # the O- sterically open, so we can park the charged ARG41 next to the O- for
    # a clean salt bridge WITHOUT it clashing into the rest of the ligand.
    lig_base = []
    lig_base.append([1.5, 0.0, 0.0])                  # atom 0: O- at the +x tip
    for a in range(1, L):
        lig_base.append([1.5 - 1.6 * a, 0.0, 0.0])    # mild atoms trailing in -x

    res_base = []
    # ARG41: placed ~3.6 A from the carboxylate O- (atom 0 at x=1.5), i.e. right
    # at the LJ minimum rmin_ij = rmin_half(ARG)+rmin_half(O-) = 1.91+1.66 = 3.57 A
    # so the van der Waals term is near zero (NOT on the repulsive r^-12 wall) and
    # the strong +/- Coulomb attraction dominates -> a clean SALT-BRIDGE hot spot.
    res_base.append([1.5 + 3.6, 0.0, 0.0])
    # LEU88: a fat neutral residue parked ~4.1 A off-axis from the O- atom, near
    # its own LJ minimum rmin_ij = 2.10+1.66 = 3.76 A -> a favourable vdW (shape)
    # contact with zero electrostatics -> a clean van der Waals hot spot.
    res_base.append([1.5, 4.1, 0.0])
    # Background residues: scattered on a shell 9-14 A out (mostly beyond strong
    # interaction, some inside the 12 A cutoff but contributing little).
    for m in range(2, M):
        ang = 2.0 * math.pi * (m - 2) / max(1, M - 2)
        rad = 9.0 + 4.0 * rng.random()
        res_base.append([rad * math.cos(ang), rad * math.sin(ang), 2.0 * rng.random()])

    def jitter(p, amp):
        # Small Gaussian wiggle to emulate thermal motion across frames.
        return [c + rng.gauss(0.0, amp) for c in p]

    # --- Emit the file ----------------------------------------------------
    lines = []
    lines.append("# SYNTHETIC protein-ligand system for project 2.23 (NOT real data).")
    lines.append("# Generated by scripts/make_synthetic.py -- see data/README.md.")
    lines.append("# Header: F M L cutoff(Angstrom)")
    lines.append(f"{F} {M} {L} {args.cutoff:g}")
    lines.append("# Residue params: name charge eps rmin_half born")
    for r in res:
        lines.append(f"{r['name']} {r['q']:g} {r['eps']:g} {r['rmin']:g} {r['born']:g}")
    lines.append("# Ligand params: charge eps rmin_half born")
    for a in lig:
        lines.append(f"{a['q']:g} {a['eps']:g} {a['rmin']:g} {a['born']:g}")
    for f in range(F):
        lines.append(f"# frame {f}: M residue 'x y z' lines then L ligand 'x y z' lines")
        amp = 0.05 if f == 0 else 0.20            # frame 0 unjittered for a clean anchor
        for m in range(M):
            x, y, z = jitter(res_base[m], amp) if f > 0 else res_base[m]
            lines.append(f"{x:.4f} {y:.4f} {z:.4f}")
        for a in range(L):
            x, y, z = jitter(lig_base[a], amp) if f > 0 else lig_base[a]
            lines.append(f"{x:.4f} {y:.4f} {z:.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  (F={F}, M={M}, L={L}; SYNTHETIC)")


if __name__ == "__main__":
    main()
