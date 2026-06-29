#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic FEP/TI sample input
# ---------------------------------------------------------------------------
# Project 1.5 : Free Energy Perturbation / Thermodynamic Integration
#
# WHY THIS EXISTS
#   Real FEP benchmark sets (Merck/OpenFE, FEP+, PDBbind) are full molecular
#   systems that need a complete MD engine + force field -- far beyond this
#   reduced-scope teaching model, and several are license- or registration-gated
#   (see data/README.md and scripts/download_data.*). So the committed sample is
#   a tiny, clearly-SYNTHETIC config for our 1-D harmonic alchemical model, whose
#   free-energy difference has a CLOSED FORM the demo verifies against.
#
#   The model morphs harmonic state A (k=kA) into state B (k=kB) along a linear
#   lambda-path; the exact answer is  DeltaG = 1/2 * kT * ln(kB / kA)  (THEORY.md).
#   The default kA=1, kB=4, kT=1 gives DeltaG = 1/2 ln(4) = ln(2) ~ 0.693147 --
#   a memorable target the TI estimate should reproduce.
#
# OUTPUT FORMAT (one whitespace-separated record; see data/README.md):
#   kA x0A kB x0B kT windows equil samples step x_init
#
# USAGE
#   python scripts/make_synthetic.py                 # default sample
#   python scripts/make_synthetic.py --windows 21    # finer lambda-grid
#   python scripts/make_synthetic.py --kB 9          # DeltaG = 1/2 ln(9) ~ 1.0986
# ===========================================================================
import argparse
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "alchemy_sample.txt"


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic FEP/TI sample.")
    ap.add_argument("--kA", type=float, default=1.0, help="state-A spring constant")
    ap.add_argument("--x0A", type=float, default=0.0, help="state-A well centre")
    ap.add_argument("--kB", type=float, default=4.0, help="state-B spring constant")
    ap.add_argument("--x0B", type=float, default=1.0, help="state-B well centre")
    ap.add_argument("--kT", type=float, default=1.0, help="temperature (energy units)")
    ap.add_argument("--windows", type=int, default=11, help="number of lambda-windows")
    ap.add_argument("--equil", type=int, default=2000, help="MC burn-in steps")
    ap.add_argument("--samples", type=int, default=20000, help="MC averaged steps")
    ap.add_argument("--step", type=float, default=0.6, help="MC trial half-width")
    ap.add_argument("--x_init", type=float, default=0.0, help="chain start position")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    fields = [args.kA, args.x0A, args.kB, args.x0B, args.kT,
              args.windows, args.equil, args.samples, args.step, args.x_init]
    line = " ".join(repr(v) if isinstance(v, float) else str(v) for v in fields)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    header = ("# SYNTHETIC FEP/TI sample (Project 1.5) -- NOT real molecular data.\n"
              "# fields: kA x0A kB x0B kT windows equil samples step x_init\n")
    Path(args.out).write_text(header + line + "\n", encoding="utf-8")

    dG = 0.5 * args.kT * math.log(args.kB / args.kA)
    print(f"[make_synthetic] wrote {args.out}  (SYNTHETIC)")
    print(f"[make_synthetic] analytic DeltaG = 1/2*{args.kT}*ln({args.kB}/{args.kA}) = {dG:.6f}")


if __name__ == "__main__":
    main()
