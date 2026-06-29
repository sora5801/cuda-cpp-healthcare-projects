#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Write the SYNTHETIC SMD configuration
# ---------------------------------------------------------------------------
# Project 1.26 : Steered Molecular Dynamics (SMD)
#
# WHY THIS EXISTS
#   Real SMD inputs are full-atom molecular systems (PDB structures, force-field
#   topologies) that need NAMD/GROMACS/OpenMM to simulate -- far beyond a tiny
#   committed sample. This project teaches the SMD *method* (pull along a
#   coordinate, accumulate non-equilibrium work, recover ΔG with Jarzynski's
#   equality) on a reduced 1-D model whose free-energy landscape is KNOWN
#   analytically, so the demo is self-checking. This script writes that model's
#   parameters. The data is SYNTHETIC and labeled synthetic everywhere
#   (CLAUDE.md §8); it is not a real molecule.
#
# THE ENGINEERED SAMPLE (PATTERNS.md §6: embed a known answer)
#   The PMF is a tilted double well U(xi)=A*((xi-xa)(xi-xb))^2/(xb-xa)^2 + slope*xi.
#   Because the pull runs well-to-well (xi0->xi_end), the TRUE end-to-end free
#   energy is exactly ΔG = U(xi_end)-U(xi0) = slope*(xi_end-xi0) (the quartic
#   term vanishes at both wells). With slope=-12 and a 1 nm pull, ΔG = -12 kJ/mol.
#   The pull is deliberately slow + soft so Jarzynski's estimate recovers that
#   value within ~1 kJ/mol while the naive mean work <W> stays clearly biased --
#   the whole teaching point.
#
# OUTPUT (data/README.md format), one whitespace-separated line:
#   xi0 xi_end n_traj steps dt k_spring v_pull gamma kT pmf_A pmf_xa pmf_xb
#   pmf_slope seed
#
# USAGE
#   python scripts/make_synthetic.py
#   python scripts/make_synthetic.py --n-traj 65536      # tighter Jarzynski
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "smd_config.txt"


def main():
    ap = argparse.ArgumentParser(description="Write the synthetic SMD configuration.")
    ap.add_argument("--xi0", type=float, default=0.0, help="bound-state coordinate (nm)")
    ap.add_argument("--xi-end", type=float, default=1.0, help="unbound-state coordinate (nm)")
    ap.add_argument("--n-traj", type=int, default=8192, help="number of SMD trajectories")
    ap.add_argument("--steps", type=int, default=25000, help="Langevin steps per pull")
    ap.add_argument("--dt", type=float, default=0.002, help="timestep (ps)")
    ap.add_argument("--k-spring", type=float, default=2000.0, help="spring stiffness (kJ/mol/nm^2)")
    ap.add_argument("--v-pull", type=float, default=0.02, help="pull velocity (nm/ps)")
    ap.add_argument("--gamma", type=float, default=500.0, help="Langevin friction")
    ap.add_argument("--kT", type=float, default=2.4943, help="kB*T at 300 K (kJ/mol)")
    ap.add_argument("--pmf-A", type=float, default=25.0, help="PMF barrier scale (kJ/mol)")
    ap.add_argument("--pmf-xa", type=float, default=0.0, help="bound-well centre (nm)")
    ap.add_argument("--pmf-xb", type=float, default=1.0, help="unbound-well centre (nm)")
    ap.add_argument("--pmf-slope", type=float, default=-12.0, help="tilt (kJ/mol/nm); sets true ΔG")
    ap.add_argument("--seed", type=int, default=20240626, help="base RNG seed")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    # Sanity: the dummy atom should travel exactly xi_end-xi0 over the pull, i.e.
    # v_pull*steps*dt == (xi_end-xi0). We do not force it, but we warn if it is off
    # so a learner editing parameters keeps the geometry self-consistent.
    travel = args.v_pull * args.steps * args.dt
    span = args.xi_end - args.xi0
    note = "" if abs(travel - span) < 1e-9 else \
        f"  [warn] v*steps*dt={travel:g} != xi_end-xi0={span:g}"

    # Floats use %g; the integer counts and the 64-bit seed use %d so the seed is
    # NOT mangled into scientific notation (which the C++ loader reads as a float).
    fields = [
        f"{args.xi0:g}", f"{args.xi_end:g}", f"{args.n_traj:d}", f"{args.steps:d}",
        f"{args.dt:g}", f"{args.k_spring:g}", f"{args.v_pull:g}", f"{args.gamma:g}",
        f"{args.kT:g}", f"{args.pmf_A:g}", f"{args.pmf_xa:g}", f"{args.pmf_xb:g}",
        f"{args.pmf_slope:g}", f"{args.seed:d}",
    ]
    line = " ".join(fields)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(line + "\n", encoding="utf-8")

    true_dg = args.pmf_slope * span
    print(f"[make_synthetic] wrote {args.out}  ({args.n_traj} trajectories x "
          f"{args.steps} steps; true dG = {true_dg:g} kJ/mol; SYNTHETIC){note}")


if __name__ == "__main__":
    main()
