#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Write the defibrillation-sweep sample file
# ---------------------------------------------------------------------------
# Project 6.19 : Defibrillation & High-Voltage Shock Simulation
#
# This project's "data" is the SIMULATION SETUP for a defibrillation-threshold
# (DFT) sweep on a 1-D monodomain cardiac cable with FitzHugh-Nagumo ionic
# kinetics. There is no patient data here at all -- everything is SYNTHETIC and
# clearly labelled as such (see data/README.md and CLAUDE.md section 8). Real
# defibrillation data (electrograms, whole-heart bidomain meshes) lives behind
# the sources documented in data/README.md and scripts/download_data.*.
#
# WHY THESE DEFAULTS
#   They are hand-tuned so the sweep is INTERPRETABLE and reproduces a textbook
#   DFT curve: the weakest shocks leave a residual travelling wave (defibrillation
#   FAILS), and above the threshold the tissue is reset to rest (SUCCESS). With
#   the defaults the recovered DFT is amplitude 0.15 (index 3). Editing any value
#   and re-running the program will shift the curve -- a good exercise.
#
# OUTPUT FORMAT (three lines; parsed by load_sweep in src/reference_cpu.cpp):
#   line 1: ncell nsteps dt dx D a eps gamma
#   line 2: initial_excited shock_start shock_len biphasic success_thresh
#   line 3: namp  a0 a1 a2 ...            (namp ascending shock amplitudes)
#
# STABILITY: forward-Euler diffusion needs dt <= dx^2/(2 D). With dx=1, D=0.6
# that limit is 0.833; the default dt=0.1 is comfortably inside it.
#
# USAGE
#   python scripts/make_synthetic.py
#   python scripts/make_synthetic.py --biphasic 1 --shock-len 20
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "defib_sweep.txt"

# The ascending shock-amplitude ladder swept to locate the threshold.
DEFAULT_AMPS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.60, 1.00]


def main():
    ap = argparse.ArgumentParser(description="Write the defibrillation DFT-sweep sample file.")
    # --- cable + FHN parameters (line 1) ---
    ap.add_argument("--ncell", type=int, default=100, help="cells along the 1-D cable")
    ap.add_argument("--nsteps", type=int, default=2000, help="time steps to integrate")
    ap.add_argument("--dt", type=float, default=0.1, help="time step (must be <= dx^2/(2D))")
    ap.add_argument("--dx", type=float, default=1.0, help="cell spacing (dimensionless)")
    ap.add_argument("--D", type=float, default=0.6, help="diffusion / gap-junction coupling")
    ap.add_argument("--a", type=float, default=0.13, help="FHN excitation threshold (0<a<1)")
    ap.add_argument("--eps", type=float, default=0.008, help="recovery rate (small = slow)")
    ap.add_argument("--gamma", type=float, default=1.0, help="recovery coupling")
    # --- initial condition + shock protocol (line 2) ---
    ap.add_argument("--initial-excited", type=int, default=30,
                    help="left-hand cells started excited (seeds the wave)")
    ap.add_argument("--shock-start", type=int, default=800, help="step the shock turns on")
    ap.add_argument("--shock-len", type=int, default=10, help="shock duration (steps)")
    ap.add_argument("--biphasic", type=int, default=0, choices=(0, 1),
                    help="0 = monophasic (sample default), 1 = biphasic")
    ap.add_argument("--success-thresh", type=float, default=0.01,
                    help="residual activity below which a shock counts as success")
    ap.add_argument("--amps", type=float, nargs="+", default=DEFAULT_AMPS,
                    help="ascending shock amplitudes to sweep")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    # Guard the diffusion stability limit up front so we never write a sample the
    # solver will reject (the solver checks the same thing and errors out).
    dt_max = (args.dx * args.dx) / (2.0 * args.D)
    if args.dt > dt_max:
        raise SystemExit(f"[make_synthetic] dt={args.dt} exceeds stability limit "
                         f"dx^2/(2D)={dt_max:.4f}; lower dt or D.")

    amps = sorted(args.amps)   # ascending, so find_dft returns the weakest success
    lines = [
        f"{args.ncell} {args.nsteps} {args.dt} {args.dx} {args.D} {args.a} {args.eps} {args.gamma}",
        f"{args.initial_excited} {args.shock_start} {args.shock_len} {args.biphasic} {args.success_thresh}",
        f"{len(amps)} " + " ".join(f"{v:g}" for v in amps),
    ]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[make_synthetic] wrote {args.out}")
    print(f"  cable: ncell={args.ncell} nsteps={args.nsteps} dt={args.dt} dx={args.dx} D={args.D}")
    print(f"  FHN:   a={args.a} eps={args.eps} gamma={args.gamma}")
    print(f"  shock: start={args.shock_start} len={args.shock_len} "
          f"{'biphasic' if args.biphasic else 'monophasic'}  thresh={args.success_thresh}")
    print(f"  swept {len(amps)} amplitudes: {amps}")


if __name__ == "__main__":
    main()
