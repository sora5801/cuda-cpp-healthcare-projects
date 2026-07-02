#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic sample dataset
# ---------------------------------------------------------------------------
# Project 6.10 : Systems-Biology ODE/SDE Network Solver
#
# WHY THIS EXISTS
#   Real systems-biology models live in curated repositories as SBML files
#   (BioModels, VCell) whose parsing/JIT is a whole engineering project of its
#   own (see README "Prior art"). For a self-contained TEACHING demo we instead
#   generate a clearly-SYNTHETIC parameter-sweep config for the canonical
#   3-gene "repressilator" circuit (Elowitz & Leibler, Nature 2000). This keeps
#   the focus on the GPU batch-ODE pattern, not on SBML plumbing. Synthetic data
#   is always LABELED synthetic (see data/README.md).
#
#   Output layout (whitespace-separated; consumed by src/reference_cpu.cpp):
#     alpha0 beta dt steps na nn alpha_lo alpha_hi n_lo n_hi  m0 m1 m2 p0 p1 p2
#   The C++ loader reads bare numeric tokens with operator>>, so this file must
#   contain ONLY numbers (no comment lines).
#
# USAGE
#   python scripts/make_synthetic.py                 # default 6x6 sweep
#   python scripts/make_synthetic.py --na 32 --nn 32 # a denser sweep
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "ensemble_params.txt"


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic repressilator ensemble config.")
    ap.add_argument("--alpha0", type=float, default=1.0, help="basal (leaky) transcription")
    ap.add_argument("--beta", type=float, default=5.0, help="protein/mRNA decay ratio")
    ap.add_argument("--dt", type=float, default=0.05, help="RK4 timestep (mRNA lifetimes)")
    ap.add_argument("--steps", type=int, default=4000, help="number of timesteps")
    ap.add_argument("--na", type=int, default=6, help="alpha sweep points")
    ap.add_argument("--nn", type=int, default=6, help="Hill-coefficient sweep points")
    ap.add_argument("--alpha-lo", type=float, default=10.0)
    ap.add_argument("--alpha-hi", type=float, default=260.0)
    ap.add_argument("--n-lo", type=float, default=1.0)
    ap.add_argument("--n-hi", type=float, default=3.0)
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    # Shared initial state [m0 m1 m2 p0 p1 p2]: a small asymmetric seed (m0=1)
    # so the ring is not stuck at the symmetric fixed point.
    s0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    header = [args.alpha0, args.beta, args.dt, args.steps, args.na, args.nn,
              args.alpha_lo, args.alpha_hi, args.n_lo, args.n_hi]

    def fmt(v):
        return f"{int(v)}" if isinstance(v, int) else f"{v:g}"

    tokens = [fmt(v) for v in header] + [f"{v:g}" for v in s0]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(" ".join(tokens) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  "
          f"({args.na}x{args.nn} = {args.na * args.nn} members; SYNTHETIC repressilator)")


if __name__ == "__main__":
    main()
