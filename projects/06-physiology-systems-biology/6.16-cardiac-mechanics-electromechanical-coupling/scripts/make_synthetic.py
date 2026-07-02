#!/usr/bin/env python3
# =============================================================================
# scripts/make_synthetic.py  --  Generate the synthetic heart-ensemble sample
# -----------------------------------------------------------------------------
# Project 6.16 : Cardiac Mechanics & Electromechanical Coupling
#
# WHAT THIS WRITES
#   A single whitespace-separated text file describing the BASELINE physiology
#   of a lumped 0-D left ventricle (time-varying elastance + Windkessel) plus
#   the (contractility x afterload) sweep grid. The C++ program (load_ensemble
#   in src/reference_cpu.cpp) reads exactly this layout. The "data" is a MODEL
#   CONFIGURATION, not measured signals -- it is entirely SYNTHETIC and
#   illustrative (physiological ballpark, not fitted to any patient).
#   NOT FOR CLINICAL USE.
#
#   The blocks (in file order, matching load_ensemble):
#     1. activation:   t_activate
#     2. calcium:      Ca_rest Ca_amp tau_rise tau_decay
#     3. cross-bridge: Tref_baseline Ca50 nH k_xb
#     4. chamber:      Emin V0
#     5. valves:       R_ao R_mv P_ven
#     6. windkessel:   R_sys_baseline C_art P_art_dias
#     7. integration:  dt steps_per_beat n_beats
#     8. sweep:        nT nR Tref_lo Tref_hi R_lo R_hi
#
# USAGE
#   python make_synthetic.py                       # default 6x6 = 36 hearts
#   python make_synthetic.py --nT 32 --nR 32       # 1024 hearts
#   python make_synthetic.py -o ../data/sample/heart_ensemble.txt
#
# There is no randomness: the sweep is a deterministic grid, so the sample (and
# therefore the demo's expected_output.txt) is fully reproducible.
# =============================================================================
import argparse
import os


def main():
    ap = argparse.ArgumentParser(description="Write the synthetic heart-ensemble config.")
    ap.add_argument("-o", "--out",
                    default=os.path.join(os.path.dirname(__file__),
                                         "..", "data", "sample", "heart_ensemble.txt"),
                    help="output path")
    ap.add_argument("--nT", type=int, default=6, help="# contractility values")
    ap.add_argument("--nR", type=int, default=6, help="# afterload values")
    args = ap.parse_args()

    # ---- Baseline physiology (lumped, illustrative) -----------------------
    # Timing: cell activates 20 ms into the cycle.
    t_activate = 20.0
    # Calcium transient: rest 0.1 uM, peak +1.0 uM, fast rise / slower decay.
    Ca_rest, Ca_amp, tau_rise, tau_decay = 0.1, 1.0, 20.0, 60.0
    # Cross-bridge: baseline Tref (overridden by sweep), Hill Ca50/n, rate k_xb.
    Tref_baseline, Ca50, nH, k_xb = 2.5, 0.6, 4.0, 0.015
    # Chamber elastance: diastolic Emin, unloaded volume V0.
    Emin, V0 = 0.08, 15.0
    # Valves: aortic + mitral resistances, venous filling pressure.
    R_ao, R_mv, P_ven = 0.015, 0.35, 10.0
    # Windkessel: baseline R_sys (overridden), compliance, diastolic floor.
    R_sys_baseline, C_art, P_art_dias = 1.2, 2.0, 75.0
    # Integration: dt=0.1 ms, 8000 steps/beat (=800 ms ~= 75 bpm), 10 beats
    # (9 warm-up beats to reach the limit cycle + 1 recorded).
    dt, steps_per_beat, n_beats = 0.1, 8000, 10
    # Sweep: contractility 1.5..3.5 mmHg/mL, afterload R_sys 0.7..2.2 mmHg*ms/mL.
    Tref_lo, Tref_hi = 1.5, 3.5
    R_lo, R_hi = 0.7, 2.2

    lines = [
        f"{t_activate}",
        f"{Ca_rest} {Ca_amp} {tau_rise} {tau_decay}",
        f"{Tref_baseline} {Ca50} {nH} {k_xb}",
        f"{Emin} {V0}",
        f"{R_ao} {R_mv} {P_ven}",
        f"{R_sys_baseline} {C_art} {P_art_dias}",
        f"{dt} {steps_per_beat} {n_beats}",
        f"{args.nT} {args.nR} {Tref_lo} {Tref_hi} {R_lo} {R_hi}",
    ]

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[make_synthetic] wrote {out}")
    print(f"[make_synthetic] {args.nT} x {args.nR} = {args.nT * args.nR} virtual hearts "
          f"(SYNTHETIC; not for clinical use).")


if __name__ == "__main__":
    main()
