#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic backbone sample
# ---------------------------------------------------------------------------
# Project 2.10 : Protein Design / Inverse Folding Inference
#
# WHY THIS EXISTS
#   Real backbones come from the PDB / CATH (see data/README.md), but to keep the
#   demo OFFLINE and the result INTERPRETABLE we generate a clearly-SYNTHETIC
#   toy "protein" whose answer we can reason about (PATTERNS.md sec 6). Synthetic
#   data is always LABELED synthetic, here and in data/README.md.
#
#   THE CONSTRUCTION (a graded, recover-able toy):
#     * Residues are placed at a RANGE of depths from a central core to an
#       exposed surface, so their neighbour counts (burial) span the whole scale
#       -- not just two extremes. That graded burial is what lets a *diverse*
#       sequence be designed (a quadratic-well model, see src/inverse_folding.h).
#     * The NATIVE residue at each position is then chosen to (mostly) FIT that
#       position's burial: we compute the burial here with the SAME contact rule
#       the C++ code uses, pick the amino acid whose preferred burial is closest,
#       and then MUTATE a fraction of positions to a random residue. The mutation
#       fraction makes "native sequence recovery" land near a realistic, < 100%
#       value (a real native sequence is not perfectly optimal for one toy energy)
#       -- so the demo teaches the recovery metric honestly.
#
#   Everything is driven by a FIXED random seed, so the committed sample (and thus
#   expected_output.txt) is byte-stable.
#
#   NOTE: the burial rule and the preferred-burial table below MUST mirror
#   src/inverse_folding.h. They are duplicated here ONLY so the data generator is
#   self-contained Python; the C++ build never reads this file. If you change the
#   model in inverse_folding.h, regenerate the sample.
#
# OUTPUT FORMAT (see data/README.md):
#   line 1 : "<L>"
#   next L : "<x> <y> <z> <native_one_letter>"
#
# USAGE
#   python scripts/make_synthetic.py                  # writes the committed sample
#   python scripts/make_synthetic.py --shells 8 --per 6 --mutate 0.2 --seed 2026
# ===========================================================================
import argparse
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "backbone_sample.txt"

# --- These constants MIRROR src/inverse_folding.h (see NOTE above) ----------
CONTACT_RADIUS = 10.0          # angstrom; two Calpha within this are "in contact"
# Canonical 20-aa order: Ala Arg Asn Asp Cys Gln Glu Gly His Ile Leu Lys Met
#                        Phe Pro Ser Thr Trp Tyr Val
AA_CODES = "ARNDCQEGHILKMFPSTWYV"
#                  A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V
PREFERRED_BURIAL = [14,  3,  7,  4, 20,  6,  2, 11,  8, 23, 22,  4, 18, 24, 10, 12, 13, 19, 16, 23]


def neighbor_count(coords, i):
    """Count residues within CONTACT_RADIUS of residue i (matches the C++ rule)."""
    xi, yi, zi = coords[i]
    r2 = CONTACT_RADIUS * CONTACT_RADIUS
    c = 0
    for j, (xj, yj, zj) in enumerate(coords):
        if j == i:
            continue
        dx, dy, dz = xi - xj, yi - yj, zi - zj
        if dx * dx + dy * dy + dz * dz <= r2:
            c += 1
    return c


def best_aa_for_burial(nbr):
    """Index of the amino acid whose preferred burial is closest to `nbr`
    (lowest index wins ties) -- the same argmax the C++ design makes."""
    best_k, best_d = 0, abs(nbr - PREFERRED_BURIAL[0])
    for k in range(1, len(PREFERRED_BURIAL)):
        d = abs(nbr - PREFERRED_BURIAL[k])
        if d < best_d:
            best_k, best_d = k, d
    return best_k


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic backbone sample.")
    ap.add_argument("--shells", type=int, default=6, help="number of concentric depth shells")
    ap.add_argument("--per", type=int, default=5, help="residues per OUTER shell (inner shells denser)")
    ap.add_argument("--mutate", type=float, default=0.25,
                    help="fraction of natives randomized (controls recovery < 100%%)")
    ap.add_argument("--seed", type=int, default=2026, help="RNG seed (fixed => stable sample)")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    rng = random.Random(args.seed)     # deterministic: same seed => same file

    # --- Place residues on concentric shells from deep core to far surface ---
    # Shell 0 is the tightly packed core (small radius => many neighbours); the
    # outermost shell is the exposed surface (large radius => few neighbours).
    # Inner shells are packed DENSER (more residues) and spaced CLOSER so the core
    # residues genuinely exceed BURIAL_THRESHOLD (>=16 contacts), giving a real
    # buried/exposed split; outer shells fan out (radius grows quadratically) so
    # surface residues are well separated. A golden-angle spiral spreads residues
    # evenly within each shell.
    coords = []
    for s in range(args.shells):
        radius = 3.0 + 4.0 * s                       # 3 A core, +4 A per shell
        count = max(3, args.per + 2 * (args.shells - 1 - s))  # denser core shells
        for k in range(count):
            phi = math.acos(1 - 2 * (k + 0.5) / count)        # polar angle
            theta = math.pi * (1 + 5 ** 0.5) * (k + s)        # golden azimuth
            x = radius * math.sin(phi) * math.cos(theta)
            y = radius * math.sin(phi) * math.sin(theta)
            z = radius * math.cos(phi)
            coords.append((x, y, z))

    L = len(coords)

    # --- Choose the native residue per position from its burial, then mutate --
    natives = []
    for i in range(L):
        nbr = neighbor_count(coords, i)
        if rng.random() < args.mutate:
            natives.append(rng.randrange(len(AA_CODES)))   # a random "non-optimal" native
        else:
            natives.append(best_aa_for_burial(nbr))        # a fitting native

    # --- Emit the file -------------------------------------------------------
    lines = [str(L)]
    for (x, y, z), aa in zip(coords, natives):
        # 3 decimals of angstrom precision is plenty and keeps the file tiny.
        lines.append(f"{x:.3f} {y:.3f} {z:.3f} {AA_CODES[aa]}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  "
          f"(L={L}: {args.shells} shells x {args.per}; mutate={args.mutate}; "
          f"seed={args.seed}; SYNTHETIC)")


if __name__ == "__main__":
    main()
