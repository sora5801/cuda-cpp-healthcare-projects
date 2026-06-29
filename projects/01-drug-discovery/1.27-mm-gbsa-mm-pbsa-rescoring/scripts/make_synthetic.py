#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic complex + trajectory
# ---------------------------------------------------------------------------
# Project 1.27 : MM-GBSA / MM-PBSA Rescoring
#
# WHY THIS EXISTS
#   Real protein-ligand trajectories (PDBbind, AMBER tutorials) are large and/or
#   carry licenses, so we ship a TINY, clearly-SYNTHETIC stand-in that exercises
#   exactly the same code path. The synthetic geometry is engineered to be
#   INTERPRETABLE (docs/PATTERNS.md §6): a small charged receptor pocket plus a
#   ligand that drifts outward over the "trajectory", so the per-snapshot binding
#   energy climbs toward zero as the ligand unbinds -- a result a learner can
#   sanity-check by eye.
#
#   Everything is generated with plain arithmetic (no RNG) so the file -- and
#   therefore demo/expected_output.txt -- is byte-for-byte reproducible.
#
# FILE FORMAT (matches load_complex() in src/reference_cpu.cpp; see data/README):
#   line 1:        R  L  S  minus_TdS
#   next R lines:  receptor atoms   "x y z  q  sigma eps  born"
#   next S*L lines: ligand atoms, grouped by snapshot (frame 0's L atoms first)
#   '#' starts a comment; blank lines are ignored.
#
# USAGE
#   python scripts/make_synthetic.py                 # writes the committed sample
#   python scripts/make_synthetic.py --snapshots 64  # a bigger synthetic problem
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent           # the project folder
OUT = ROOT / "data" / "sample" / "complex_sample.txt"


def build(receptor_atoms: int, ligand_atoms: int, snapshots: int, minus_TdS: float):
    """Return (header_line, receptor_lines, ligand_lines) as lists of strings.
    Deterministic: positions/charges are simple closed-form functions of the
    atom index and snapshot index -- no randomness, so output is reproducible."""
    # --- receptor: a line of alternating partial charges (a crude pocket) ---
    # Atom j sits at x = 3*j on the x-axis; charges alternate sign and shrink so
    # the pocket has a net field the ligand responds to.
    receptor = []
    for j in range(receptor_atoms):
        x = 3.0 * j
        q = (-0.6 if j % 2 == 0 else 0.5) * (1.0 - 0.05 * j)   # alternating, decaying
        # sigma/eps are generic carbon-like LJ params; born ~ a typical heavy atom.
        receptor.append(f"{x:7.3f} {0.0:6.3f} {0.0:6.3f}  {q:7.4f}  "
                        f"{3.40:5.3f} {0.10:5.3f}  {2.00:5.3f}   # receptor atom {j}")

    # --- ligand snapshots: the ligand drifts +z each frame (unbinding) ------
    # Ligand atom i starts near the pocket (x = 1.5 + 3*i) and the whole ligand
    # is displaced by d = 3.0 + 1.5*s angstroms along z in snapshot s. As d grows
    # the ligand leaves the pocket, so the favorable interaction weakens.
    ligand = []
    for s in range(snapshots):
        d = 3.0 + 1.5 * s
        for i in range(ligand_atoms):
            x = 1.5 + 3.0 * i
            z = d + 1.0 * i
            q = (0.55 if i % 2 == 0 else -0.45)
            ligand.append(f"{x:7.3f} {0.0:6.3f} {z:7.3f}  {q:7.4f}  "
                          f"{3.25:5.3f} {0.12:5.3f}  {1.80:5.3f}   # frame {s} ligand atom {i}")
    header = f"{receptor_atoms} {ligand_atoms} {snapshots} {minus_TdS:g}"
    return header, receptor, ligand


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic MM-GBSA complex sample.")
    ap.add_argument("--receptor", type=int, default=3, help="receptor atom count")
    ap.add_argument("--ligand", type=int, default=2, help="ligand atom count")
    ap.add_argument("--snapshots", type=int, default=6, help="number of MD snapshots")
    ap.add_argument("--minus-tds", type=float, default=8.0, help="constant -T*dS [kcal/mol]")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    header, receptor, ligand = build(args.receptor, args.ligand, args.snapshots, args.minus_tds)

    lines = []
    lines.append("# SYNTHETIC MM-GBSA complex -- NOT real structural data (Project 1.27).")
    lines.append("# Format: header 'R L S minus_TdS', then R receptor atoms, then S*L")
    lines.append("# ligand atoms grouped by snapshot. Fields: x y z  q  sigma eps  born.")
    lines.append(header)
    lines.append("# --- receptor atoms (rigid across snapshots) ---")
    lines.extend(receptor)
    lines.append("# --- ligand snapshots (S frames x L atoms), ligand drifts +z (unbinds) ---")
    lines.extend(ligand)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  "
          f"(R={args.receptor}, L={args.ligand}, S={args.snapshots}; SYNTHETIC)")


if __name__ == "__main__":
    main()
