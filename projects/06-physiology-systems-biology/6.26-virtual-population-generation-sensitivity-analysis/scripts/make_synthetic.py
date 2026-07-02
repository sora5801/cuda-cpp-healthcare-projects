#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Write the virtual-population / Sobol config
# ---------------------------------------------------------------------------
# Project 6.26 : Virtual Population Generation & Sensitivity Analysis
#
# The "data" is the STUDY SETUP, not per-patient rows: the oral dose, the
# plausible uniform range for each of the k=4 uncertain PK parameters
# (ka, CL, V, F), the integration horizon, and the Saltelli base sample size N.
# The virtual patients themselves are generated deterministically inside the
# program by a Halton quasi-random sequence (src/vpop.h), so the whole study is
# reproducible without shipping any patient table.
#
# OUTPUT (data/README.md format), whitespace-separated:
#   dose
#   ka_lo ka_hi
#   CL_lo CL_hi
#   V_lo  V_hi
#   F_lo  F_hi
#   t_end steps
#   N seed
#
# USAGE
#   python scripts/make_synthetic.py
#   python scripts/make_synthetic.py --N 16384
#
# NOTE: the committed sample is SYNTHETIC and illustrative (not fitted to any
# real drug). See data/README.md for provenance and real-data pointers.
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "vpop_config.txt"


def main():
    ap = argparse.ArgumentParser(description="Write the Sobol virtual-population config.")
    ap.add_argument("--dose", type=float, default=100.0, help="oral dose (mg)")
    # Uniform prior ranges for each uncertain parameter (physiologically plausible,
    # illustrative). ka: absorption; CL: clearance; V: volume; F: bioavailability.
    ap.add_argument("--ka_lo", type=float, default=0.5, help="ka lower (1/h)")
    ap.add_argument("--ka_hi", type=float, default=2.0, help="ka upper (1/h)")
    ap.add_argument("--CL_lo", type=float, default=3.0, help="CL lower (L/h)")
    ap.add_argument("--CL_hi", type=float, default=8.0, help="CL upper (L/h)")
    ap.add_argument("--V_lo", type=float, default=20.0, help="V lower (L)")
    ap.add_argument("--V_hi", type=float, default=50.0, help="V upper (L)")
    ap.add_argument("--F_lo", type=float, default=0.6, help="F lower (fraction)")
    ap.add_argument("--F_hi", type=float, default=1.0, help="F upper (fraction)")
    ap.add_argument("--t_end", type=float, default=72.0, help="AUC horizon (h)")
    ap.add_argument("--steps", type=int, default=720, help="trapezoid steps over [0,t_end]")
    ap.add_argument("--N", type=int, default=4096, help="Saltelli base sample size")
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    # One field per logical line for readability; the C++ loader skips whitespace.
    text = (
        f"{args.dose:g}\n"
        f"{args.ka_lo:g} {args.ka_hi:g}\n"
        f"{args.CL_lo:g} {args.CL_hi:g}\n"
        f"{args.V_lo:g} {args.V_hi:g}\n"
        f"{args.F_lo:g} {args.F_hi:g}\n"
        f"{args.t_end:g} {args.steps}\n"
        f"{args.N} {args.seed}\n"
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text, encoding="utf-8")

    # Analytic sanity numbers the program should reproduce (AUC = F*Dose/CL):
    #   Sobol variance is driven almost entirely by CL and F because ka and V drop
    #   out of the closed form. That is the built-in teaching check.
    total = args.N * (4 + 2)
    print(f"[make_synthetic] wrote {args.out}")
    print(f"[make_synthetic] N={args.N} -> {total} model evaluations "
          f"(N*(k+2) with k=4)")
    print(f"[make_synthetic] AUC = F*Dose/CL depends only on F,CL -> expect "
          f"Sobol S(CL)+S(F) ~ 1, S(ka)+S(V) ~ 0")


if __name__ == "__main__":
    main()
