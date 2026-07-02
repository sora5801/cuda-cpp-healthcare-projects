#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Write the virtual-patient cohort configuration
# ---------------------------------------------------------------------------
# Project 6.23 : Glucose-Insulin Dynamics & Artificial Pancreas
#
# WHAT THE "DATA" IS
#   The input is not a measured signal -- it is the SETUP for an in-silico trial:
#   the fixed Bergman-minimal-model constants + meal + PID controller settings,
#   plus the two-parameter SWEEP that defines the cohort of virtual patients
#   (insulin sensitivity via the insulin-action gain p3, and glucose
#   effectiveness p1). The program derives every patient's parameters from this
#   grid, so the whole cohort is reproducible from one line of numbers.
#
#   This is SYNTHETIC and labeled synthetic everywhere. The parameter values are
#   illustrative teaching values loosely in the range of published Bergman
#   minimal-model fits; they are NOT fitted to any real subject and the output is
#   NOT clinically valid (see data/README.md, README §Limitations).
#
# OUTPUT (data/README.md format), one whitespace-separated line of 26 values:
#   p2 n Gb Ib VG VI  meal_D meal_Ag meal_k meal_t  G_target Kp Ki Kd
#   u_basal u_max control_dt  G0 dt steps  nSI nSG  p3_lo p3_hi  p1_lo p1_hi
#
# USAGE
#   python scripts/make_synthetic.py                    # default 32x32 = 1024 patients
#   python scripts/make_synthetic.py --nSI 64 --nSG 64  # 4096 patients
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "cohort_params.txt"


def main():
    ap = argparse.ArgumentParser(description="Write the artificial-pancreas cohort config.")
    # --- Fixed Bergman minimal-model constants (units in src/bergman.h) ---
    ap.add_argument("--p2", type=float, default=0.025, help="remote-insulin decay [1/min]")
    ap.add_argument("--n",  type=float, default=0.14,  help="plasma insulin clearance [1/min]")
    ap.add_argument("--Gb", type=float, default=110.0, help="basal glucose [mg/dL]")
    ap.add_argument("--Ib", type=float, default=12.0,  help="basal insulin [uU/mL]")
    ap.add_argument("--VG", type=float, default=117.0, help="glucose distribution volume [dL]")
    ap.add_argument("--VI", type=float, default=25.0,  help="insulin distribution volume (scales pump)")
    # --- Meal disturbance (two-exponential gastric emptying) ---
    ap.add_argument("--meal-D",  type=float, default=50000.0, help="meal carbs [mg glucose] (~50 g)")
    ap.add_argument("--meal-Ag", type=float, default=0.8,     help="meal bioavailability")
    ap.add_argument("--meal-k",  type=float, default=0.05,    help="gut-absorption rate [1/min] (peak ~20 min)")
    ap.add_argument("--meal-t",  type=float, default=30.0,    help="meal start time [min]")
    # --- PID controller + insulin pump ---
    ap.add_argument("--G-target", type=float, default=110.0, help="glucose set-point [mg/dL]")
    ap.add_argument("--Kp", type=float, default=1.2,   help="PID proportional gain")
    ap.add_argument("--Ki", type=float, default=0.010, help="PID integral gain")
    ap.add_argument("--Kd", type=float, default=12.0,  help="PID derivative gain")
    ap.add_argument("--u-basal", type=float, default=1.0,  help="basal infusion")
    ap.add_argument("--u-max",   type=float, default=80.0, help="pump maximum infusion")
    ap.add_argument("--control-dt", type=float, default=5.0, help="controller period [min]")
    # --- Integration ---
    ap.add_argument("--G0",    type=float, default=140.0, help="initial glucose [mg/dL]")
    ap.add_argument("--dt",    type=float, default=0.5,   help="RK4 timestep [min]")
    ap.add_argument("--steps", type=int,   default=960,   help="steps (run = steps*dt min = 8 h)")
    # --- Cohort sweep ---
    ap.add_argument("--nSI", type=int, default=32, help="number of insulin-sensitivity values")
    ap.add_argument("--nSG", type=int, default=32, help="number of glucose-effectiveness values")
    ap.add_argument("--p3-lo", type=float, default=1.0e-5, help="insulin-action gain low (SI=p3/p2)")
    ap.add_argument("--p3-hi", type=float, default=4.0e-5, help="insulin-action gain high")
    ap.add_argument("--p1-lo", type=float, default=0.018, help="glucose effectiveness (SG) low")
    ap.add_argument("--p1-hi", type=float, default=0.028, help="glucose effectiveness (SG) high")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    vals = [args.p2, args.n, args.Gb, args.Ib, args.VG, args.VI,
            args.meal_D, args.meal_Ag, args.meal_k, args.meal_t,
            args.G_target, args.Kp, args.Ki, args.Kd,
            args.u_basal, args.u_max, args.control_dt,
            args.G0, args.dt, args.steps,
            args.nSI, args.nSG, args.p3_lo, args.p3_hi, args.p1_lo, args.p1_hi]
    # ':g' keeps integers integer-looking and floats compact; steps/nSI/nSG stay ints.
    line = " ".join(str(v) if isinstance(v, int) else f"{v:g}" for v in vals)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(line + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  "
          f"({args.nSI * args.nSG} virtual patients, "
          f"{int(args.steps * args.dt)} min run; SI in "
          f"[{args.p3_lo / args.p2:.4f}, {args.p3_hi / args.p2:.4f}]; SYNTHETIC)")


if __name__ == "__main__":
    main()
