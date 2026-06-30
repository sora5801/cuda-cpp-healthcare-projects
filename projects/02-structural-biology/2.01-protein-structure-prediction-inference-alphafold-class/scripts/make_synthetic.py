#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the tiny synthetic attention sample
# ---------------------------------------------------------------------------
# Project 2.1 : Protein Structure Prediction Inference (AlphaFold-class)
#               REDUCED-SCOPE TEACHING VERSION.
#
# WHAT THIS WRITES  (data/sample/attention_sample.txt)
#   One self-attention head's three input matrices Q, K, V, each [L x d], in the
#   text format documented in data/README.md:
#       line 1 : "<L> <d>"
#       then   : L rows of d floats for Q, then L rows for K, then L rows for V.
#   '#' lines are comments (the C++ loader skips them).
#
# WHY THESE EXACT NUMBERS  (must match make_synthetic_problem() in src/main.cu)
#   The data is SYNTHETIC and engineered to have a known, verifiable answer
#   (PATTERNS.md sec 6). Residue r places a large PEAK value in feature channel
#   (r % d) of its Q and K vectors, plus a small deterministic ramp everywhere.
#   Because the peaks line up only when query i meets key i, every residue
#   attends most strongly to ITSELF -- the identity-like baseline of self-
#   attention -- which is exactly what the demo prints and checks. The V matrix
#   stores the residue index (r+1) in channel 0 so the mixed output is readable.
#
#   NOT BIOLOGICAL DATA. These are not real protein features, MSAs, or learned
#   weights -- just a reproducible toy that exercises the kernel. See the README
#   "Limitations & honesty" and data/README.md.
#
# USAGE
#   python scripts/make_synthetic.py                       # default L=6, d=32
#   python scripts/make_synthetic.py --L 6 --out data/sample/attention_sample.txt
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent             # the project folder
# d MUST equal D_MODEL in src/attention_core.h (the loader rejects a mismatch).
D_MODEL = 32
PEAK = 3.0


def build(L: int, d: int):
    """Return (Q, K, V) as lists of L rows of d floats, IDENTICAL to the C++
    make_synthetic_problem(). Keeping the two in lockstep means the committed
    sample reproduces the built-in fallback exactly."""
    Q, K, V = [], [], []
    for r in range(L):
        q_row, k_row, v_row = [], [], []
        for c in range(d):
            ramp = 0.01 * float((r * d + c) % 7)             # small non-degenerate ramp
            peak = PEAK if c == (r % d) else 0.0             # residue r's identity channel
            q_row.append(ramp + peak)
            k_row.append(ramp + peak)
            v_row.append(float(r + 1) if c == 0 else ramp)   # residue index in channel 0
        Q.append(q_row)
        K.append(k_row)
        V.append(v_row)
    return Q, K, V


def fmt(x: float) -> str:
    """Compact, loader-friendly float formatting (7 significant digits). All our
    values are short decimals, so this round-trips well within the 1e-5 tol."""
    return f"{x:.7g}"


def main():
    ap = argparse.ArgumentParser(description="Write the synthetic self-attention sample.")
    ap.add_argument("--L", type=int, default=6, help="number of residues (sequence length)")
    ap.add_argument("--d", type=int, default=D_MODEL, help="feature width (must equal D_MODEL=32)")
    ap.add_argument("--out", default=str(ROOT / "data" / "sample" / "attention_sample.txt"))
    args = ap.parse_args()

    if args.d != D_MODEL:
        raise SystemExit(f"[make_synthetic] d must equal D_MODEL={D_MODEL} (got {args.d})")

    Q, K, V = build(args.L, args.d)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write(f"{args.L} {args.d}\n")
        f.write("# SYNTHETIC self-attention sample (project 2.1). NOT real protein data.\n")
        f.write("# Q matrix: L rows of d floats (queries), row-major\n")
        for row in Q:
            f.write(" ".join(fmt(x) for x in row) + "\n")
        f.write("# K matrix: L rows of d floats (keys)\n")
        for row in K:
            f.write(" ".join(fmt(x) for x in row) + "\n")
        f.write("# V matrix: L rows of d floats (values)\n")
        for row in V:
            f.write(" ".join(fmt(x) for x in row) + "\n")
    print(f"[make_synthetic] wrote {out}  (L={args.L}, d={args.d}, SYNTHETIC)")


if __name__ == "__main__":
    main()
