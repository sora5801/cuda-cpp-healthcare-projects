#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic ensemble sample
# ---------------------------------------------------------------------------
# Project 6.2 : Whole-Heart Digital Twin   (REDUCED-SCOPE TEACHING VERSION)
#
# WHY THIS EXISTS
#   The real datasets a whole-heart twin is built from (UK Biobank CMR, ACDC,
#   Visible Human) require credentials/registration and cannot be redistributed
#   (see data/README.md). This project does not need patient images at all: its
#   input is a small text CONFIG describing an ENSEMBLE of virtual hearts
#   (a contractility sweep) plus a synthetic clinical target stroke volume.
#   This script writes that config so the demo runs fully offline. The data is
#   SYNTHETIC and labeled synthetic everywhere.
#
#   Output layout (whitespace-separated, one logical record; see data/README.md):
#     n emax_lo emax_hi dt_ms beats target_sv bcl_ms E_min V0 Rp C_art
#   where:
#     n         : number of virtual hearts in the ensemble
#     emax_lo/hi: contractility (systolic elastance) sweep range [mmHg/mL]
#     dt_ms     : RK4 timestep [ms]
#     beats     : cardiac cycles to simulate (transient wash-out + measurement)
#     target_sv : synthetic clinical target stroke volume the twin is fit to [mL]
#     bcl_ms    : basic cycle length / beat period [ms]  (800 ms = 75 bpm)
#     E_min     : diastolic elastance [mmHg/mL]
#     V0        : unstressed ventricular volume [mL]
#     Rp        : peripheral (systemic) resistance [mmHg*s/mL]
#     C_art     : arterial compliance [mL/mmHg]
#
# USAGE
#   python scripts/make_synthetic.py                 # default committed sample
#   python scripts/make_synthetic.py --n 256         # bigger ensemble
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent           # the project folder
OUT = ROOT / "data" / "sample" / "heart_ensemble.txt"


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic heart-ensemble sample.")
    ap.add_argument("--n", type=int, default=12, help="number of virtual hearts")
    ap.add_argument("--emax-lo", type=float, default=1.2, help="lowest contractility [mmHg/mL]")
    ap.add_argument("--emax-hi", type=float, default=3.4, help="highest contractility [mmHg/mL]")
    ap.add_argument("--dt-ms", type=float, default=0.10, help="RK4 timestep [ms]")
    ap.add_argument("--beats", type=int, default=6, help="cardiac cycles to simulate")
    ap.add_argument("--target-sv", type=float, default=70.0, help="target stroke volume [mL]")
    ap.add_argument("--bcl-ms", type=float, default=800.0, help="beat period [ms]")
    ap.add_argument("--e-min", type=float, default=0.06, help="diastolic elastance [mmHg/mL]")
    ap.add_argument("--v0", type=float, default=10.0, help="unstressed volume [mL]")
    ap.add_argument("--rp", type=float, default=1.0, help="peripheral resistance [mmHg*s/mL]")
    ap.add_argument("--c-art", type=float, default=1.6, help="arterial compliance [mL/mmHg]")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    # One whitespace-separated record in the loader's exact field order. `g`
    # formatting keeps the file tiny and human-readable.
    fields = [args.n, args.emax_lo, args.emax_hi, args.dt_ms, args.beats,
              args.target_sv, args.bcl_ms, args.e_min, args.v0, args.rp, args.c_art]
    line = " ".join(f"{v:g}" for v in fields)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(line + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  "
          f"(n={args.n}, Emax {args.emax_lo}..{args.emax_hi} mmHg/mL; SYNTHETIC)")


if __name__ == "__main__":
    main()
