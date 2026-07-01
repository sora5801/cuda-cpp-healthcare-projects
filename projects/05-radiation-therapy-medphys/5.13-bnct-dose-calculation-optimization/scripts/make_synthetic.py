#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic BNCT sample dataset
# ---------------------------------------------------------------------------
# Project 5.13 : BNCT Dose Calculation & Optimization (reduced-scope teaching MC)
#
# WHY THIS EXISTS
#   Real BNCT benchmark inputs (IAEA cases, ENDF/B-VIII cross-section libraries,
#   clinical CT) require registration and cannot be redistributed here (see
#   data/README.md). So we ship a tiny, clearly-SYNTHETIC parameter file that
#   drives the reduced-scope 1-D two-group Monte Carlo. It is labeled synthetic
#   everywhere and makes NO clinical claim.
#
#   The values are physically MOTIVATED (order-of-magnitude realistic macroscopic
#   cross sections for soft tissue at thermal energy) so the demo is
#   interpretable -- boron's giant capture cross section dominates the thermal
#   capture rate, which is the whole point of BNCT.
#
# FILE LAYOUT (one whitespace-separated line; parsed by reference_cpu.cpp):
#   L n_bins Sig_s_fast p_thermalize Sig_a_B Sig_a_N Sig_a_H Sig_s_th
#   Q_boron_keV Q_nitro_keV Q_gamma_keV Q_fast_keV n_histories seed gray_per_keV
#
# USAGE
#   python scripts/make_synthetic.py                 # default sample
#   python scripts/make_synthetic.py --histories 1000000 --seed 7
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "bnct_params.txt"

# ---------------------------------------------------------------------------
# Default macroscopic cross sections (Sigma = N * sigma, units 1/cm), chosen to
# be order-of-magnitude realistic for soft tissue at a thermal energy of
# 0.025 eV (2200 m/s). References for the microscopic sigmas: ^10B sigma_a ~
# 3837 b, ^14N sigma_a ~ 1.83 b, ^1H sigma_a ~ 0.332 b (thermal). We fold in
# representative number densities + a therapeutic ~30 ppm ^10B loading so the
# boron capture term dominates. THESE ARE TEACHING VALUES, NOT A LIBRARY.
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    L=10.0,             # 10 cm tissue slab (cm)
    n_bins=20,          # 0.5 cm depth resolution
    Sig_s_fast=0.90,    # fast-neutron scatter Sigma_s (1/cm) -- mostly on H
    p_thermalize=0.25,  # P(a fast scatter thermalizes the neutron)
    Sig_a_B=0.070,      # ^10B capture Sigma_a (1/cm) at ~30 ppm 10B (dominant)
    Sig_a_N=0.005,      # ^14N capture Sigma_a (1/cm)
    Sig_a_H=0.021,      # ^1H  capture Sigma_a (1/cm)
    Sig_s_th=1.50,      # thermal scatter Sigma_s (1/cm) -- the random walk
    Q_boron_keV=2310,   # effective boron reaction energy deposited locally (keV)
    Q_nitro_keV=626,    # ^14N(n,p)^14C proton energy (keV)
    Q_gamma_keV=2224,   # ^1H(n,gamma)^2H capture-gamma energy (keV)
    Q_fast_keV=500,     # credited recoil-proton energy per fast scatter (keV)
    n_histories=200000, # neutron histories (small so the demo runs in ~ms)
    seed=12345,         # RNG base seed (fixed -> deterministic sample)
    gray_per_keV=1.0e-12,  # teaching scale keV-quanta -> Gy (NOT clinical)
)


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic BNCT sample.")
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k}", type=type(v), default=v)
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = vars(ap.parse_args())
    out = args.pop("out")

    order = ["L", "n_bins", "Sig_s_fast", "p_thermalize",
             "Sig_a_B", "Sig_a_N", "Sig_a_H", "Sig_s_th",
             "Q_boron_keV", "Q_nitro_keV", "Q_gamma_keV", "Q_fast_keV",
             "n_histories", "seed", "gray_per_keV"]
    fields = " ".join(f"{args[k]:g}" if isinstance(args[k], float) else str(args[k])
                      for k in order)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    header = ("# SYNTHETIC BNCT parameters (project 5.13) -- teaching only, NOT clinical.\n"
              "# L n_bins Sig_s_fast p_thermalize Sig_a_B Sig_a_N Sig_a_H Sig_s_th "
              "Q_boron_keV Q_nitro_keV Q_gamma_keV Q_fast_keV n_histories seed gray_per_keV\n")
    Path(out).write_text(header + fields + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {out}  (SYNTHETIC BNCT params; "
          f"histories={args['n_histories']}, seed={args['seed']})")


if __name__ == "__main__":
    main()
