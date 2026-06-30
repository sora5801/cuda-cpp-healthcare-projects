#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic HP-lattice sample
# ---------------------------------------------------------------------------
# Project 2.7 : Monte Carlo Protein Structure Sampling (HP lattice model)
#
# WHY THIS EXISTS
#   Real folding benchmarks (CASP/PDB structures, Dunbrack rotamer libraries)
#   are full 3-D coordinate sets that this reduced 2-D HP teaching model does
#   not consume, and some are license/registration gated. So we ship a tiny,
#   clearly-SYNTHETIC HP sequence that the demo can fold offline. The sequence
#   is a well-known short HP benchmark-style chain whose folded state buries
#   several H-H contacts, so the result is meaningful and verifiable.
#
# THE FILE FORMAT (see data/README.md and src/reference_cpu.cpp::load_mc_problem)
#   line 1:  n sweeps n_replicas t_min t_max seed
#   line 2:  the HP sequence (n characters from {H,P})
#
#   These EXACT parameters mirror the program's built-in fallback in main.cu, so
#   running with or without the file gives the same deterministic result -- which
#   is what demo/expected_output.txt encodes.
#
# USAGE
#   python scripts/make_synthetic.py                 # writes data/sample/hp_problem.txt
#   python scripts/make_synthetic.py --replicas 512  # bigger ensemble
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "hp_problem.txt"

# An 18-residue synthetic HP chain (10 H, 8 P). SYNTHETIC -- not a real protein.
# It is short enough to fold quickly yet long enough to form a compact core with
# multiple buried H-H contacts, which makes the demo's "best energy" meaningful.
DEFAULT_SEQUENCE = "HPHPPHHPHHPHHPPHPH"


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic HP-lattice MC sample.")
    ap.add_argument("--sequence", default=DEFAULT_SEQUENCE,
                    help="HP sequence (characters H/P)")
    ap.add_argument("--sweeps", type=int, default=600, help="MC sweeps per replica")
    ap.add_argument("--replicas", type=int, default=256, help="independent walkers")
    ap.add_argument("--t-min", type=float, default=0.30, help="coldest replica T")
    ap.add_argument("--t-max", type=float, default=3.00, help="hottest replica T")
    ap.add_argument("--seed", type=int, default=20260628, help="base RNG seed")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    seq = args.sequence.upper()
    if any(c not in "HP" for c in seq):
        raise SystemExit("sequence must contain only H and P characters")
    n = len(seq)
    if n < 2:
        raise SystemExit("sequence must have at least 2 residues")

    header = f"{n} {args.sweeps} {args.replicas} {args.t_min:g} {args.t_max:g} {args.seed}"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(header + "\n" + seq + "\n", encoding="utf-8")
    n_h = sum(1 for c in seq if c == "H")
    print(f"[make_synthetic] wrote {args.out}")
    print(f"[make_synthetic]   n={n} ({n_h} H), sweeps={args.sweeps}, "
          f"replicas={args.replicas}, T in [{args.t_min:g}, {args.t_max:g}]; SYNTHETIC")


if __name__ == "__main__":
    main()
