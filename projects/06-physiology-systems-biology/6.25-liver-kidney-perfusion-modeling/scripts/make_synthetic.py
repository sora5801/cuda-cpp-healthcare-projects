#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic lobule sample
# ---------------------------------------------------------------------------
# Project 6.25 : Liver & Kidney Perfusion Modeling
#
# WHY THIS EXISTS
#   The real inputs (Human Protein Atlas zonal enzyme expression, HMDB liver
#   metabolite levels, Open Systems Pharmacology PBPK parameters) are either
#   license-restricted or require curation into a model. So we ship a tiny,
#   clearly-SYNTHETIC lobule config that lets demo/run_demo run OFFLINE. The
#   numbers are physiologically PLAUSIBLE but invented -- never clinical.
#
# THE MODEL (see ../THEORY.md)
#   A liver lobule = `nsin` parallel sinusoids. Each is a 1-D plug-flow tube of
#   length L (periportal x=0 -> centrilobular x=L). A drug enters at C_in and is
#   cleared by wall enzymes at a Michaelis-Menten rate Vmax(x)*C/(Km+C), where
#   Vmax ramps LINEARLY from Vmax_pp (periportal) to Vmax_cl (centrilobular) --
#   the "metabolic zonation" of the liver. Perfusion is heterogeneous: inlet
#   blood velocity is swept linearly from v_lo (slow) to v_hi (fast).
#
#   Units: length mm, concentration uM, velocity mm/s, Vmax uM/s (consistent so
#   the transport balance v*dC/dx [uM/s] = -R [uM/s] holds).
#
#   We keep C_in << Km so the reaction is near the FIRST-ORDER regime; that lets
#   main.cu cross-check the numerics against a closed-form exponential washout.
#
# FILE LAYOUT (whitespace, one logical record):
#   L C_in Km Vmax_pp Vmax_cl nseg   nsin v_lo v_hi
#
# USAGE
#   python scripts/make_synthetic.py                 # writes data/sample/lobule.txt
#   python scripts/make_synthetic.py --nsin 65536    # bigger lobule (more threads)
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "lobule.txt"


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic liver-lobule sample.")
    # Physical constants (physiologically plausible, SYNTHETIC):
    ap.add_argument("--L", type=float, default=0.5, help="sinusoid length (mm)")
    ap.add_argument("--C_in", type=float, default=1.0, help="inlet drug concentration (uM)")
    ap.add_argument("--Km", type=float, default=50.0, help="Michaelis constant (uM)")
    ap.add_argument("--Vmax_pp", type=float, default=8.0, help="periportal max clearance (uM/s)")
    ap.add_argument("--Vmax_cl", type=float, default=2.0, help="centrilobular max clearance (uM/s)")
    ap.add_argument("--nseg", type=int, default=200, help="RK4 spatial steps along the sinusoid")
    # Ensemble (one thread per sinusoid):
    ap.add_argument("--nsin", type=int, default=4096, help="number of parallel sinusoids")
    ap.add_argument("--v_lo", type=float, default=0.2, help="slowest inlet velocity (mm/s)")
    ap.add_argument("--v_hi", type=float, default=1.0, help="fastest inlet velocity (mm/s)")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    # One line, in the exact order load_lobule() reads.
    record = (f"{args.L:g} {args.C_in:g} {args.Km:g} {args.Vmax_pp:g} {args.Vmax_cl:g} "
              f"{args.nseg:d}   {args.nsin:d} {args.v_lo:g} {args.v_hi:g}")

    header = ("# SYNTHETIC liver-lobule perfusion config for project 6.25 (NOT clinical).\n"
              "# Fields: L C_in Km Vmax_pp Vmax_cl nseg   nsin v_lo v_hi\n"
              "#   L[mm] C_in[uM] Km[uM] Vmax_pp[uM/s] Vmax_cl[uM/s] nseg[-]  nsin[-] v_lo[mm/s] v_hi[mm/s]\n")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(header + record + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  (nsin={args.nsin}; SYNTHETIC)")


if __name__ == "__main__":
    main()
