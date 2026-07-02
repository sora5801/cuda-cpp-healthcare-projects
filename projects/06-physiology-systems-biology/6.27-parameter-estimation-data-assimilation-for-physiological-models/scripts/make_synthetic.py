#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic EnKF config for 6.27
# ---------------------------------------------------------------------------
# Project 6.27 : Parameter Estimation & Data Assimilation for Physiological Models
#
# WHAT THIS WRITES
#   A single-line, whitespace-separated config file consumed by the program
#   (data/README.md documents the exact field order). The "data" is a SYNTHETIC
#   twin-experiment setup: a KNOWN true patient (R_true, C_true) whose noisy
#   aortic-pressure waveform the C++ program regenerates internally and then tries
#   to recover with an Ensemble Kalman Filter. Nothing here is from a real patient;
#   it is engineered so the demo result is interpretable -- the filter should
#   recover R_true, C_true to a few percent from a deliberately-wrong prior.
#
# WHY a script (not hand-typed numbers): reproducibility, and easy rescaling for
#   the exercises (bigger ensemble, more windows).
#
# USAGE
#   python scripts/make_synthetic.py                              # default sample
#   python scripts/make_synthetic.py --ensemble 1024 --windows 80
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "enkf_config.txt"


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic Windkessel-EnKF config (project 6.27).")
    # --- Ensemble & assimilation schedule ---
    ap.add_argument("--ensemble", type=int, default=256, help="ensemble size m")
    ap.add_argument("--windows", type=int, default=40, help="number of observation windows (n_obs)")
    ap.add_argument("--dt", type=float, default=0.005, help="RK4 sub-step (s)")
    ap.add_argument("--substeps", type=int, default=16, help="RK4 sub-steps per observation window")
    # --- Inflow waveform Q(t) (mL/s): a half-sine ejection per beat ---
    ap.add_argument("--period", type=float, default=0.8, help="cardiac cycle T (s); 0.8 = 75 bpm")
    ap.add_argument("--t_sys", type=float, default=0.3, help="systolic ejection duration (s)")
    ap.add_argument("--q_peak", type=float, default=500.0, help="peak inflow (mL/s)")
    # --- TRUE patient parameters (the target the filter must recover) ---
    ap.add_argument("--r_true", type=float, default=1.0, help="true peripheral resistance (mmHg*s/mL)")
    ap.add_argument("--c_true", type=float, default=1.5, help="true arterial compliance (mL/mmHg)")
    ap.add_argument("--p0", type=float, default=80.0, help="initial aortic pressure (mmHg)")
    # --- Noise ---
    ap.add_argument("--obs_noise", type=float, default=1.0, help="measurement noise std (mmHg)")
    # --- Prior (deliberately off, so recovery is visible) ---
    ap.add_argument("--r_prior", type=float, default=1.4, help="prior guess for R (mmHg*s/mL)")
    ap.add_argument("--c_prior", type=float, default=1.0, help="prior guess for C (mL/mmHg)")
    ap.add_argument("--logr_std", type=float, default=0.3, help="prior std of log R")
    ap.add_argument("--logc_std", type=float, default=0.3, help="prior std of log C")
    ap.add_argument("--seed", type=int, default=20260701, help="RNG seed (reproducible)")
    ap.add_argument("--out", default=str(OUT), help="output path")
    a = ap.parse_args()

    # Field order MUST match load_config() in src/reference_cpu.cpp and the table
    # in data/README.md. One line, whitespace separated. Use repr() for floats so
    # the values round-trip exactly.
    fields = [
        a.ensemble, a.windows, repr(a.dt), a.substeps,
        repr(a.period), repr(a.t_sys), repr(a.q_peak),
        repr(a.r_true), repr(a.c_true), repr(a.p0),
        repr(a.obs_noise),
        repr(a.r_prior), repr(a.c_prior), repr(a.logr_std), repr(a.logc_std),
        a.seed,
    ]
    line = " ".join(str(f) for f in fields) + "\n"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(line, encoding="ascii", newline="\n")
    print(f"[make_synthetic] wrote {a.out}  (SYNTHETIC twin experiment)")
    print(f"  {line.strip()}")


if __name__ == "__main__":
    main()
