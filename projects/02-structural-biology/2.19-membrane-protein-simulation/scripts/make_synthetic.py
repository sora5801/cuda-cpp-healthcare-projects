#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic membrane-patch sample
# ---------------------------------------------------------------------------
# Project 2.19 : Membrane Protein Simulation   (reduced-scope teaching version)
#
# WHY THIS EXISTS
#   The real datasets (MemProtMD, GPCRdb, OPM -- see data/README.md) are full
#   atomistic structures that need force-field setup tools (CHARMM-GUI) and are
#   far beyond this teaching model. So the demo runs on a TINY, clearly-SYNTHETIC
#   coarse-grained membrane patch generated here -- no download, no credentials.
#   Everything this produces is SYNTHETIC and labeled as such.
#
#   The simulation BUILDS the bilayer geometry itself (see build_system() in
#   src/reference_cpu.cpp); this file only writes the PARAMETER line that the
#   loader reads. The format (one whitespace-separated record) is:
#
#     n_lipids n_prot box_x box_y sigma rcut k_bond r_bond dt steps temp gamma seed
#     eHH eHT eHP eTT eTP ePP            # 6 unique LJ well depths (symmetric 3x3)
#
#   Reduced MD units throughout (length in sigma, energy in epsilon). The LJ
#   well-depth matrix encodes the hydrophobic effect: TAIL-TAIL attraction (eTT)
#   is the strongest, which is what holds the two leaflets together as a bilayer.
#
# USAGE
#   python scripts/make_synthetic.py                       # default tiny patch
#   python scripts/make_synthetic.py --n-lipids 32 --steps 400
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "membrane_sample.txt"


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic membrane patch sample.")
    ap.add_argument("--n-lipids", type=int, default=18, help="number of lipids (3 beads each)")
    ap.add_argument("--n-prot", type=int, default=5, help="number of protein beads (column)")
    ap.add_argument("--box", type=float, default=6.0, help="periodic box size in x and y (sigma)")
    ap.add_argument("--steps", type=int, default=200, help="MD steps to run")
    ap.add_argument("--dt", type=float, default=0.005, help="timestep (reduced units)")
    ap.add_argument("--temp", type=float, default=0.6, help="thermostat kT (reduced units)")
    ap.add_argument("--gamma", type=float, default=1.0, help="Langevin friction (1/time)")
    ap.add_argument("--seed", type=int, default=20240617, help="master RNG seed")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    # Fixed physical constants of the model (reduced units).
    sigma = 1.0          # bead diameter
    rcut = 2.5           # LJ cutoff (the standard 2.5 sigma)
    k_bond = 30.0        # harmonic bond stiffness (stiff rod-like lipids)
    r_bond = 1.0         # bond rest length (= sigma)

    # LJ well-depth matrix (eps), symmetric. Larger -> stronger attraction.
    #   HEAD-HEAD : mild (heads are polar; modest cohesion)
    #   HEAD-TAIL : weak  (oil-and-water: heads avoid tails -> drives ordering)
    #   HEAD-PROT : mild  (protein surface near the head region)
    #   TAIL-TAIL : STRONG (the hydrophobic effect that builds the bilayer core)
    #   TAIL-PROT : strong-ish (a transmembrane protein likes the oily core)
    #   PROT-PROT : mild  (keeps the protein column cohesive)
    eHH, eHT, eHP = 0.5, 0.2, 0.5
    eTT, eTP, ePP = 1.0, 0.8, 0.6

    # Integer-valued fields are written WITHOUT %g (which would turn a big seed
    # into scientific notation and lose digits); only true floats use %g.
    scalar_str = (
        f"{args.n_lipids} {args.n_prot} {args.box:g} {args.box:g} {sigma:g} {rcut:g} "
        f"{k_bond:g} {r_bond:g} {args.dt:g} {args.steps} {args.temp:g} {args.gamma:g} "
        f"{args.seed}"
    )
    eps = [eHH, eHT, eHP, eTT, eTP, ePP]

    lines = [
        "# SYNTHETIC coarse-grained membrane patch -- Project 2.19 (NOT real data).",
        "# Format: n_lipids n_prot box_x box_y sigma rcut k_bond r_bond dt steps temp gamma seed",
        scalar_str,
        "# LJ well depths (symmetric 3x3): eHH eHT eHP eTT eTP ePP",
        " ".join(f"{v:g}" for v in eps),
    ]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    n_beads = 3 * args.n_lipids + args.n_prot
    print(f"[make_synthetic] wrote {args.out}  "
          f"(n_lipids={args.n_lipids}, n_prot={args.n_prot}, n_beads={n_beads}; SYNTHETIC)")


if __name__ == "__main__":
    main()
