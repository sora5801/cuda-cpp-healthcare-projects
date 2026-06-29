#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate a synthetic IFP-clustering dataset
# ---------------------------------------------------------------------------
# Project 1.33 : Interaction Fingerprinting & Binding-Mode Clustering
#
# WHAT IT BUILDS  (clearly labeled SYNTHETIC -- no real protein/ligand)
#   * A fixed binding pocket of 24 residues placed on a small lattice, each
#     tagged with which interaction chemistries it can make (h-bond / aromatic /
#     ionic; every residue can make a hydrophobic contact).
#   * P ligand poses drawn from K distinct BINDING MODES. A "mode" is an anchor
#     point inside the pocket; its poses are that anchor plus small Gaussian
#     jitter. Because different anchors sit near different residues, each mode
#     produces a DISTINCT interaction fingerprint -- so clustering the IFPs
#     should rediscover the K planted modes (the demo reports the purity).
#
#   The geometry uses the SAME Angstrom cutoffs as src/ifp.h (4.5 hydrophobic,
#   3.5 h-bond, 4.0 aromatic/ionic), so the planted separation is real, not
#   cosmetic. Real interaction fingerprints come from docking poses or MD frames
#   on actual complexes (PDBbind / KLIFS) -- see scripts/download_data.* and
#   data/README.md.
#
# OUTPUT FORMAT (must match load_dataset() in src/reference_cpu.cpp):
#   line 1                : "P K"
#   next 24 lines         : "x y z can_hbond can_aromatic can_ionic"
#   next P lines          : "x y z has_donor has_aromatic has_charge true_mode"
#
# USAGE
#   python scripts/make_synthetic.py                 # default tiny committed sample
#   python scripts/make_synthetic.py --per-mode 200  # bigger (for timing studies)
# ===========================================================================
import argparse
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "ifp_sample.txt"

NUM_RESIDUES = 24          # MUST equal NUM_RESIDUES in src/ifp.h

# ---------------------------------------------------------------------------
# The pocket: 24 residues on a 3-D grid (a 6x4 sheet at z=0, ~3.2 A spacing),
# each with a chemistry profile. The mix of capabilities is what lets different
# binding modes leave different fingerprints.
#   Tuple: (can_hbond, can_aromatic, can_ionic). Hydrophobic is implicit (all).
# ---------------------------------------------------------------------------
RESIDUE_CHEM = [
    (1, 0, 0), (0, 1, 0), (1, 0, 1), (0, 0, 0),   # row 0
    (1, 1, 0), (0, 0, 1), (1, 0, 0), (0, 1, 0),   # row 1
    (1, 0, 1), (0, 0, 0), (1, 1, 0), (0, 0, 1),   # row 2
    (1, 0, 0), (0, 1, 0), (1, 0, 1), (0, 0, 0),   # row 3
    (1, 1, 0), (0, 0, 1), (1, 0, 0), (0, 1, 0),   # row 4
    (1, 0, 1), (0, 0, 0), (1, 1, 0), (0, 0, 1),   # row 5
]
GRID_COLS = 4
GRID_SPACING = 3.2   # Angstrom between neighboring residues


def residue_xyz(idx):
    """Place residue idx on the sheet: row = idx // 4, col = idx % 4, z = 0."""
    row = idx // GRID_COLS
    col = idx % GRID_COLS
    return (col * GRID_SPACING, row * GRID_SPACING, 0.0)


# ---------------------------------------------------------------------------
# Binding modes: each is an anchor (x, y, z) placed just ABOVE a cluster of
# residues, plus the ligand's chemistry at that anchor. Anchors sit near
# different residue neighborhoods, so their fingerprints differ strongly.
#   Tuple: (anchor_xyz, has_donor, has_aromatic, has_charge, friendly_name)
# ---------------------------------------------------------------------------
MODES = [
    ((1.6, 1.6, 1.5), 1, 0, 0, "top-left polar"),        # near R0,R1,R4,R5
    ((8.0, 1.6, 1.5), 0, 1, 0, "top-right aromatic"),    # near R2,R3,R6,R7
    ((1.6, 14.0, 1.5), 1, 0, 1, "bottom-left ionic"),    # near R16,R17,R20,R21
    ((8.0, 14.0, 1.5), 0, 0, 1, "bottom-right charged"), # near R18,R19,R22,R23
]
SPREAD = 0.35   # Angstrom std of the per-pose jitter around an anchor


def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic IFP dataset (SYNTHETIC).")
    ap.add_argument("--per-mode", type=int, default=30, help="poses per binding mode")
    ap.add_argument("--spread", type=float, default=SPREAD, help="Gaussian jitter (Angstrom)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    K = len(MODES)

    # --- pose block (built first so we know P) ---
    pose_rows = []
    for mode_idx, (anchor, hd, ha, hc, _name) in enumerate(MODES):
        for _ in range(args.per_mode):
            x = anchor[0] + rng.gauss(0.0, args.spread)
            y = anchor[1] + rng.gauss(0.0, args.spread)
            z = anchor[2] + rng.gauss(0.0, args.spread)
            pose_rows.append(f"{x:.4f} {y:.4f} {z:.4f} {hd} {ha} {hc} {mode_idx}")
    P = len(pose_rows)

    # --- assemble the file: header, residue block, pose block ---
    lines = [f"{P} {K}"]
    for r in range(NUM_RESIDUES):
        x, y, z = residue_xyz(r)
        cb, ca, ci = RESIDUE_CHEM[r]
        lines.append(f"{x:.4f} {y:.4f} {z:.4f} {cb} {ca} {ci}")
    lines.extend(pose_rows)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  "
          f"(P={P} poses, {NUM_RESIDUES} residues, K={K} modes; SYNTHETIC)")


if __name__ == "__main__":
    main()
