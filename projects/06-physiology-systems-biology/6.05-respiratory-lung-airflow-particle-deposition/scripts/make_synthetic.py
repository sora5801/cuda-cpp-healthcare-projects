#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic sample dataset
# ---------------------------------------------------------------------------
# Project 6.5 : Respiratory / Lung Airflow & Particle Deposition
#
# WHY THIS EXISTS
#   The real inputs a whole-lung deposition study would use are patient CT scans
#   (LIDC-IDRI, COPDGene, SPIROMICS -- see data/README.md), which require
#   registration and cannot be redistributed here. So the committed demo instead
#   runs on a tiny, clearly-SYNTHETIC parameter file describing one monodisperse
#   aerosol inhaled through an idealized (Weibel-A) airway tree. This script
#   regenerates that file deterministically. Synthetic data is always LABELED
#   synthetic (CLAUDE.md section 8).
#
#   File layout (one data line, whitespace separated; '#' lines are comments):
#     d_p_microns  rho_p_kg_m3  n_gen  flow_L_per_min  n_particles  seed
#   These same six numbers are the built-in fallback in src/main.cu, so the demo
#   result is identical whether or not the file is passed on the command line.
#
# USAGE
#   python scripts/make_synthetic.py                       # default 5-um aerosol
#   python scripts/make_synthetic.py --d_p 1.0 --n 500000  # sub-micron, more MC
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "lung_params.txt"


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic lung deposition sample.")
    ap.add_argument("--d_p", type=float, default=5.0, help="particle diameter [microns]")
    ap.add_argument("--rho_p", type=float, default=1000.0, help="particle density [kg/m^3]")
    ap.add_argument("--n_gen", type=int, default=16, help="conducting-airway generations")
    ap.add_argument("--flow", type=float, default=30.0, help="inspiratory flow [L/min]")
    ap.add_argument("--n", type=int, default=200000, help="number of particle histories")
    ap.add_argument("--seed", type=int, default=12345, help="RNG seed")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    header = (
        "# Synthetic deposition experiment for project 6.5 (see data/README.md).\n"
        "# One line of six whitespace-separated fields, in human-friendly units:\n"
        "#   d_p_microns  rho_p_kg_m3  n_gen  flow_L_per_min  n_particles  seed\n"
        "# (Lines beginning with '#' are ignored by make_synthetic.py, which regenerates\n"
        "#  this file; the loader in reference_cpu.cpp reads the six numbers below.)\n"
    )
    data_line = f"{args.d_p:g} {args.rho_p:g} {args.n_gen} {args.flow:g} {args.n} {args.seed}\n"

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(header + data_line, encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  "
          f"(d_p={args.d_p} um, n_gen={args.n_gen}, n={args.n}; SYNTHETIC)")


if __name__ == "__main__":
    main()
