#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic sample dataset
# ---------------------------------------------------------------------------
# Project 2.29 : Ion Channel Gating & Permeation Simulation
#
# WHY THIS EXISTS
#   The real inputs to an ion-permeation study are huge all-atom MD trajectories
#   and umbrella-sampling PMF tables (MemProtMD, PDB structures -- see
#   data/README.md), which need registration, force fields, and gigabytes of
#   storage. For a SELF-CONTAINED, offline teaching demo we instead generate a
#   tiny, clearly-SYNTHETIC parameter file describing a single reduced 1-D pore:
#   a Gaussian potential-of-mean-force barrier plus an applied voltage. The C++
#   program then runs Brownian dynamics on it (CPU + GPU) and verifies they agree.
#
#   This is NOT a real channel and carries NO clinical meaning -- it is a
#   didactic model whose parameters are chosen so the result is interpretable:
#   a clear forward current under positive voltage and an occupancy histogram
#   that is DEPLETED at the central barrier (the selectivity-filter bottleneck).
#
# FILE LAYOUT (one whitespace-separated line, read by load_permeation_problem):
#   L n_bins U_barrier sigma q V D dt n_steps n_ions seed
#
# USAGE
#   python scripts/make_synthetic.py                 # default channel_params.txt
#   python scripts/make_synthetic.py --ions 4096     # more trajectories
#   python scripts/make_synthetic.py --voltage 0.0   # zero-field control (net~0)
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "channel_params.txt"

# Default physical parameters (reduced units: kT = 1, lengths in nm). Chosen so
# the demo is small (runs in well under a second) yet teaches the physics:
#   * U_barrier = 4 kT  -> a real desolvation barrier, depleting the pore centre.
#   * V = 4 (positive)  -> drives a clear FORWARD current (positive ion, +z).
#   * D, dt             -> diffusion step sqrt(2*D*dt) ~ 0.4 nm: coarse but stable.
DEFAULTS = dict(
    L=3.0,          # pore length (nm)
    n_bins=12,      # occupancy-histogram bins along z
    U_barrier=4.0,  # PMF barrier height at the pore centre (kT)
    sigma=0.5,      # barrier width (nm)
    q=1.0,          # ion charge (units of e); +1 for a cation (K+/Na+)
    V=4.0,          # applied transmembrane voltage (reduced e*V in kT); +V -> forward
    D=0.4,          # diffusion coefficient (nm^2 per step-unit)
    dt=0.2,         # Brownian-dynamics time step
    n_steps=2000,   # BD steps per ion trajectory
    n_ions=256,     # independent ion trajectories (the parallel work)
    seed=12345,     # base RNG seed (reproducible)
)


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic ion-channel sample.")
    ap.add_argument("--length", type=float, default=DEFAULTS["L"])
    ap.add_argument("--bins", type=int, default=DEFAULTS["n_bins"])
    ap.add_argument("--barrier", type=float, default=DEFAULTS["U_barrier"])
    ap.add_argument("--sigma", type=float, default=DEFAULTS["sigma"])
    ap.add_argument("--charge", type=float, default=DEFAULTS["q"])
    ap.add_argument("--voltage", type=float, default=DEFAULTS["V"])
    ap.add_argument("--diffusion", type=float, default=DEFAULTS["D"])
    ap.add_argument("--dt", type=float, default=DEFAULTS["dt"])
    ap.add_argument("--steps", type=int, default=DEFAULTS["n_steps"])
    ap.add_argument("--ions", type=int, default=DEFAULTS["n_ions"])
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    # The 12 fields, in the exact order load_permeation_problem expects.
    fields = [args.length, args.bins, args.barrier, args.sigma, args.charge,
              args.voltage, args.diffusion, args.dt, args.steps, args.ions, args.seed]
    line = " ".join(f"{v:g}" for v in fields)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(line + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}")
    print(f"[make_synthetic] L={args.length} nm, barrier={args.barrier} kT, "
          f"V={args.voltage}, ions={args.ions}, steps={args.steps}  (SYNTHETIC)")


if __name__ == "__main__":
    main()
