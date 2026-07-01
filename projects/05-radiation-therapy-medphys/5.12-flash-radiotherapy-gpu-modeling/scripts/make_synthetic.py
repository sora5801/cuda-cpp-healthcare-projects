#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic FLASH-ensemble sample
# ---------------------------------------------------------------------------
# Project 5.12 : FLASH Radiotherapy GPU Modeling
#
# WHY THIS EXISTS
#   Real FLASH-RT dosimetry / tumour-oxygenation datasets are credentialed or
#   not redistributable (see data/README.md). To keep the demo runnable offline
#   we generate a small, clearly-SYNTHETIC ensemble-configuration file that the
#   C++ program reads. The physics is entirely in the program; this file only
#   holds the sweep/beam parameters.
#
#   Output layout (single whitespace-separated line group), matching
#   load_ensemble() in src/reference_cpu.cpp:
#     total_dose  n_pulses  dt  conv_steps_per_gap  flash_steps_per_gap
#     relax_steps  n_po2  po2_lo  po2_hi
#
#   The defaults reproduce the interpretable FLASH signature documented in
#   THEORY.md: a pO2 sweep from 2 to 40 mmHg (tumour-hypoxic .. normal-tissue),
#   10 Gy in 10 pulses, delivered either conventionally (a long 40 ms inter-pulse
#   gap = 4000 steps of dt=1e-5 s, so O2 fully refills between pulses) or
#   FLASH/UHDR (a 10 us gap = 1 step, so O2 cannot refill). All values are
#   SYNTHETIC and for teaching only -- never clinical.
#
# USAGE
#   python scripts/make_synthetic.py                     # default sample
#   python scripts/make_synthetic.py --n-po2 16          # finer oxygen sweep
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent            # the project folder
OUT = ROOT / "data" / "sample" / "flash_ensemble.txt"


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic FLASH ensemble config.")
    ap.add_argument("--total-dose", type=float, default=10.0, help="total dose [Gy]")
    ap.add_argument("--n-pulses", type=int, default=10, help="pulses in the train")
    ap.add_argument("--dt", type=float, default=1.0e-5, help="RK4 timestep [s]")
    ap.add_argument("--conv-steps", type=int, default=4000,
                    help="RK4 sub-steps between pulses, CONVENTIONAL (large gap)")
    ap.add_argument("--flash-steps", type=int, default=1,
                    help="RK4 sub-steps between pulses, FLASH/UHDR (tiny gap)")
    ap.add_argument("--relax-steps", type=int, default=4000,
                    help="post-delivery relaxation sub-steps")
    ap.add_argument("--n-po2", type=int, default=8, help="number of oxygen levels")
    ap.add_argument("--po2-lo", type=float, default=2.0, help="min pO2 [mmHg]")
    ap.add_argument("--po2-hi", type=float, default=40.0, help="max pO2 [mmHg]")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    # One tidy, self-documenting line. %g keeps dt in scientific form.
    line = (f"{args.total_dose:g} {args.n_pulses} {args.dt:g} "
            f"{args.conv_steps} {args.flash_steps} {args.relax_steps} "
            f"{args.n_po2} {args.po2_lo:g} {args.po2_hi:g}")

    header = (
        "# SYNTHETIC FLASH-RT ensemble config (teaching only -- not clinical).\n"
        "# Fields: total_dose[Gy] n_pulses dt[s] conv_steps_per_gap "
        "flash_steps_per_gap relax_steps n_po2 po2_lo[mmHg] po2_hi[mmHg]\n"
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    # The C++ reader (load_ensemble) uses '>>' on plain doubles/ints and does NOT
    # skip comment lines, so the data file must be pure numbers -- no '#' header.
    # The field documentation lives in data/README.md instead (printed below too).
    Path(args.out).write_text(line + "\n", encoding="utf-8")
    print("[make_synthetic] wrote", args.out)
    print("[make_synthetic] (SYNTHETIC ensemble config; header for reference:)")
    print(header.rstrip())


if __name__ == "__main__":
    main()
