#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic bone-remodeling sample
# ---------------------------------------------------------------------------
# Project 6.22 : Bone Remodeling Simulation   (REDUCED-SCOPE teaching version)
#
# WHY THIS EXISTS
#   Real trabecular-bone microCT stacks (OAI, PhysioNet, BoneJ) are large and/or
#   credential-gated and cannot be redistributed here. We therefore commit a
#   TINY, clearly-SYNTHETIC parameter file so the demo runs offline with zero
#   downloads (CLAUDE.md section 8). This is NOT patient data and implies no
#   clinical validity -- it is a set of dimensionless knobs for the mechanostat
#   remodeling model implemented in src/.
#
# WHAT IT WRITES
#   A one-value-per-line text file the C++ loader (load_bone) reads in this exact
#   order:
#       nx ny remodel_steps relax_iters load load_x0 load_x1 setpoint lazy rate
#       rho_min rho_init
#   The defaults match the built-in fallback in src/main.cu (make_synthetic),
#   so the committed sample and the no-argument run produce the SAME result.
#
#   The physical story the defaults encode: a 24x16 voxel specimen with a
#   LOCALIZED load (a joint/implant contact patch) pushed in on the center of the
#   top edge, base supported. Over 60 remodeling steps the mechanostat carves an
#   oriented trabecular strut down the load path and thins the lightly-loaded
#   flanks -- so the per-column mass profile peaks under the load.
#
# USAGE
#   python scripts/make_synthetic.py                 # writes data/sample/bone_params.txt
#   python scripts/make_synthetic.py --nx 64 --ny 48 # a bigger synthetic grid
# ===========================================================================
import argparse
from pathlib import Path

# The project folder (this file lives in <project>/scripts/).
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "bone_params.txt"


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic bone-remodeling sample.")
    ap.add_argument("--nx", type=int, default=24, help="grid columns (across the load)")
    ap.add_argument("--ny", type=int, default=16, help="grid rows (along the load; row 0 = loaded top)")
    ap.add_argument("--steps", type=int, default=60, help="remodeling iterations")
    ap.add_argument("--relax", type=int, default=80, help="Jacobi sweeps per step to settle the stimulus field")
    ap.add_argument("--load", type=float, default=4.0, help="mechanical load injected under the footprint")
    ap.add_argument("--load-x0", type=int, default=10, help="first column of the loaded footprint (inclusive)")
    ap.add_argument("--load-x1", type=int, default=13, help="last column of the loaded footprint (inclusive)")
    ap.add_argument("--setpoint", type=float, default=0.55, help="homeostatic SED-per-mass target k")
    ap.add_argument("--lazy", type=float, default=0.20, help="lazy-zone half-width w (dead band)")
    ap.add_argument("--rate", type=float, default=0.05, help="remodeling gain")
    ap.add_argument("--rho-min", type=float, default=0.05, help="density floor")
    ap.add_argument("--rho-init", type=float, default=0.50, help="uniform initial density")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    # One value per line, in the loader's fixed order. repr() keeps floats exact.
    # The loader reads with operator>>, which does NOT skip '#'-comment lines, so
    # we emit a pure-number body and keep the annotated legend in data/README.md.
    fields = [
        args.nx, args.ny, args.steps, args.relax,
        args.load, args.load_x0, args.load_x1,
        args.setpoint, args.lazy, args.rate,
        args.rho_min, args.rho_init,
    ]
    lines = [f"{v!r}" if isinstance(v, float) else str(v) for v in fields]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {out_path}  "
          f"(grid {args.nx}x{args.ny}, {args.steps} steps; SYNTHETIC)")


if __name__ == "__main__":
    main()
