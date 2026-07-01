#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic slab-transport sample
# ---------------------------------------------------------------------------
# Project 5.6 : GPU Boltzmann Transport (Deterministic Dose)
#
# WHY THIS EXISTS
#   Clinical LBTE benchmarks (AAPM TG-105, IROC phantoms, Acuros XB validation
#   sets) either need credentials or cannot be redistributed. So we ship a small,
#   clearly-SYNTHETIC 1-D slab that still exercises the whole algorithm: a
#   heterogeneous "tissue / lung / tissue" stack with a localized source, exactly
#   the tissue/low-density interface where deterministic transport beats
#   pencil-beam superposition. All numbers are illustrative, NOT clinical.
#
# FILE FORMAT (see data/README.md)
#   line 1 : ncell nord width max_iter tol psi_left_bc psi_right_bc
#   then ncell lines : sigma_t sigma_s q        (one physical cell per line)
#   Units: cross-sections in 1/cm, q in particles/cm^3/s, width in cm.
#
# USAGE
#   python scripts/make_synthetic.py                 # default tiny sample
#   python scripts/make_synthetic.py --ncell 200 --nord 16
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "slab_problem.txt"


def build_slab(ncell: int, width: float):
    """Return per-cell (sigma_t, sigma_s, q) for a 3-layer slab.

    Layout along x in [0, width]:
      * outer thirds  = "tissue": denser, moderately scattering, absorbing.
      * middle third  = "lung":   low density -> small cross-sections.
      * a narrow band near the left tissue carries a fixed source (a beam entry).
    The contrast between tissue and lung is what makes the deterministic flux
    profile interesting (and is where dose engines historically disagreed).
    """
    cells = []
    for i in range(ncell):
        x = (i + 0.5) / ncell            # normalized position in [0,1]
        if 1.0 / 3.0 <= x < 2.0 / 3.0:
            # "lung": low-density -> both cross-sections shrink together.
            sigma_t, sigma_s = 0.20, 0.15
        else:
            # "tissue": higher total, strongly scattering (scatter ratio 0.8).
            sigma_t, sigma_s = 1.00, 0.80
        # A fixed isotropic source in a thin band in the first (left) tissue layer
        # -- think of it as where an external beam deposits its first interactions.
        q = 1.0 if (0.08 <= x < 0.16) else 0.0
        cells.append((sigma_t, sigma_s, q))
    return cells


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic slab problem.")
    ap.add_argument("--ncell", type=int, default=24, help="number of spatial cells")
    ap.add_argument("--nord", type=int, default=8, help="S_N order (even)")
    ap.add_argument("--width", type=float, default=6.0, help="slab thickness [cm]")
    ap.add_argument("--max-iter", type=int, default=2000, help="max source iterations")
    ap.add_argument("--tol", type=float, default=1e-10, help="convergence tolerance")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    if args.nord < 2 or args.nord % 2 != 0:
        raise SystemExit("--nord must be even and >= 2")

    cells = build_slab(args.ncell, args.width)

    lines = [f"{args.ncell} {args.nord} {args.width:g} {args.max_iter} "
             f"{args.tol:g} 0.0 0.0"]           # vacuum (zero-inflow) boundaries
    for sigma_t, sigma_s, q in cells:
        lines.append(f"{sigma_t:g} {sigma_s:g} {q:g}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  "
          f"(ncell={args.ncell}, S_{args.nord}, {args.width} cm; SYNTHETIC)")


if __name__ == "__main__":
    main()
