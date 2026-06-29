#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic sample dataset
# ---------------------------------------------------------------------------
# Project 1.9 -- ML Interatomic Potentials (Neural Network Potentials)   (template skeleton)
#
# WHY THIS EXISTS
#   Some real datasets cannot be redistributed (license) or require credentials
#   (MIMIC, UK Biobank). In those cases we still want the demo to RUN, so this
#   script deterministically generates a clearly-synthetic stand-in that matches
#   the loader's expected layout. Synthetic data is always LABELED synthetic.
#
#   Placeholder layout (SAXPY): n, a, then n x-values, then n y-values, such that
#   out = a*x + y is exact (out[i] = 12*i) so expected_output.txt is stable.
#
#   TODO(impl): regenerate this to produce the real project's synthetic input.
#
# USAGE
#   python scripts/make_synthetic.py            # writes data/sample/saxpy_sample.txt
#   python scripts/make_synthetic.py --n 1024   # bigger synthetic problem
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "saxpy_sample.txt"


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic SAXPY sample.")
    ap.add_argument("--n", type=int, default=8, help="number of elements")
    ap.add_argument("--a", type=float, default=2.0, help="scalar multiplier")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    n, a = args.n, args.a
    x = [float(i) for i in range(n)]
    y = [float(10 * i) for i in range(n)]              # out = a*x + y = 12*i (a=2)

    lines = [str(n), repr(a),
             " ".join(f"{v:g}" for v in x),
             " ".join(f"{v:g}" for v in y)]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  (n={n}, a={a}; SYNTHETIC)")


if __name__ == "__main__":
    main()
