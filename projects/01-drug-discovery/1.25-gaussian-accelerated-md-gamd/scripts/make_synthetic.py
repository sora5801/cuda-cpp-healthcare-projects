#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Write the GaMD run configuration (SYNTHETIC)
# ---------------------------------------------------------------------------
# Project 1.25 : Gaussian-Accelerated MD (GaMD)   (reduced-scope teaching version)
#
# WHAT THIS WRITES
#   The "data" for this project is the GaMD run CONFIG: the model double-well
#   potential, the Langevin thermostat, the GaMD boost parameters, and the
#   ensemble + histogram settings. The C++ program derives everything else
#   (per-walker RNG streams, the trajectory, the reweighted PMF) from this, so
#   the whole run is reproducible from these 15 numbers.
#
#   There is NO real molecular dataset here -- this is a 1-D MODEL system, clearly
#   SYNTHETIC (data/README.md says so). Real GaMD reads AMBER/NAMD topology +
#   coordinate files (prmtop/inpcrd, PDB); see scripts/download_data.* for where
#   to get those for a full study.
#
# OUTPUT FORMAT (one whitespace-separated record; see data/README.md):
#   u_barrier kT gamma_fric dt steps equil_steps
#   e_threshold v_min v_max k0
#   n_walkers x_lo x_hi n_bins seed
#
# USAGE
#   python scripts/make_synthetic.py
#   python scripts/make_synthetic.py --u-barrier 8 --n-walkers 1024 --steps 8000
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "gamd_config.txt"


def main():
    ap = argparse.ArgumentParser(description="Write the synthetic GaMD run config.")
    # --- model double well U(x) = u_barrier*(x^2-1)^2, in units of kT ---------
    ap.add_argument("--u-barrier", type=float, default=3.0,
                    help="barrier height in kT (minima at x=+/-1, barrier at x=0)")
    # --- thermodynamics + overdamped Langevin integrator ----------------------
    ap.add_argument("--kT", type=float, default=1.0, help="thermal energy k_B*T")
    ap.add_argument("--gamma", type=float, default=1.0, help="friction (overdamped drag)")
    ap.add_argument("--dt", type=float, default=0.005, help="Langevin timestep")
    ap.add_argument("--steps", type=int, default=8000, help="timesteps per walker")
    ap.add_argument("--equil", type=int, default=2000, help="leading steps not tallied")
    # --- GaMD boost: dV=0.5*k*(E-U)^2 for U<E, k=k0/(Vmax-Vmin) ----------------
    #   Pin the threshold E to the barrier top (= u_barrier) so the boost lifts the
    #   deep wells but not the (already-explored) barrier region. Vmin=0 (well
    #   bottoms), Vmax=u_barrier (barrier top).
    #   k0 is kept GENTLE (0.15): a strong boost (k0->1) accelerates sampling more
    #   but makes the 2nd-order cumulant reweighting overestimate the barrier (a
    #   real GaMD trade-off documented in THEORY §6). 0.15 recovers the PMF cleanly.
    ap.add_argument("--k0", type=float, default=0.15, help="GaMD force-constant knob (0<k0<=1)")
    # --- ensemble + PMF histogram ---------------------------------------------
    ap.add_argument("--n-walkers", type=int, default=512, help="independent walkers (= GPU threads)")
    ap.add_argument("--x-lo", type=float, default=-2.0, help="histogram lower bound on x")
    ap.add_argument("--x-hi", type=float, default=2.0, help="histogram upper bound on x")
    ap.add_argument("--n-bins", type=int, default=40, help="number of PMF bins")
    ap.add_argument("--seed", type=int, default=12345, help="base RNG seed")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    # E, Vmin, Vmax are derived from the barrier height so the harmonic-boundary
    # condition (the boost never inverts the energy ordering) holds by construction.
    e_threshold = args.u_barrier
    v_min = 0.0
    v_max = args.u_barrier

    fields = [
        f"{args.u_barrier:g}", f"{args.kT:g}", f"{args.gamma:g}", f"{args.dt:g}",
        f"{args.steps}", f"{args.equil}",
        f"{e_threshold:g}", f"{v_min:g}", f"{v_max:g}", f"{args.k0:g}",
        f"{args.n_walkers}", f"{args.x_lo:g}", f"{args.x_hi:g}", f"{args.n_bins}",
        f"{args.seed}",
    ]
    line = " ".join(fields)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(line + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  (SYNTHETIC)")
    print(f"  double well barrier = {args.u_barrier:g} kT, {args.n_walkers} walkers x "
          f"{args.steps} steps, boost k0={args.k0:g}")


if __name__ == "__main__":
    main()
