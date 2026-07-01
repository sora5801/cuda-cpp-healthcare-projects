#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic channel-flow sample
# ---------------------------------------------------------------------------
# Project 6.3 : Hemodynamics / Blood-Flow CFD   (reduced-scope teaching version)
#
# WHY THIS EXISTS
#   The real patient-specific geometries and pressure/flow waveforms named in the
#   catalog (Vascular Model Repository, MIMIC-III, UK Biobank 4D-flow MRI) are
#   either license-restricted or credential-gated and cannot be redistributed
#   (CLAUDE.md §8). So the committed sample is a small, fully SYNTHETIC channel-
#   flow *parameter set* -- not patient data. It drives the solver toward the
#   analytic Poiseuille solution, giving the demo a known, checkable answer.
#
#   This script writes ONE line of whitespace-separated numbers that the loader
#   (src/reference_cpu.cpp :: load_channel) parses in this exact order:
#
#     nx ny steps p_iters h dt rho gx nu0 nu_inf lambda n_cy a_cy
#
#   Field meanings (dimensionless "lattice" units so results are clean numbers;
#   see THEORY.md for the mapping to SI cm/s/Pa):
#     nx,ny   : grid size (x = streamwise, y = across the channel)
#     steps   : number of fractional-step time steps
#     p_iters : Jacobi iterations per pressure Poisson solve
#     h       : uniform cell spacing
#     dt      : time-step size (must satisfy the diffusive stability limit;
#               see the assertion below)
#     rho     : fluid density
#     gx      : streamwise body force (stand-in for the driving pressure gradient)
#     nu0     : Carreau-Yasuda zero-shear kinematic viscosity
#     nu_inf  : Carreau-Yasuda infinite-shear kinematic viscosity
#               (nu0 == nu_inf  =>  NEWTONIAN fluid -> matches analytic Poiseuille)
#     lambda  : Carreau-Yasuda relaxation time
#     n_cy    : power-law index (<1 => shear thinning; ignored when Newtonian)
#     a_cy    : Yasuda transition exponent
#
# USAGE
#   python scripts/make_synthetic.py                    # default Newtonian channel
#   python scripts/make_synthetic.py --nu-inf 0.03      # enable shear thinning
#   python scripts/make_synthetic.py --steps 8000       # run longer / converge more
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "channel_params.txt"


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic channel-flow sample.")
    ap.add_argument("--nx", type=int, default=32, help="streamwise grid cells")
    ap.add_argument("--ny", type=int, default=17, help="across-channel grid cells (odd => a centre row)")
    ap.add_argument("--steps", type=int, default=4000, help="time steps")
    ap.add_argument("--p-iters", type=int, default=40, help="Jacobi pressure iterations per step")
    ap.add_argument("--h", type=float, default=1.0, help="cell spacing")
    ap.add_argument("--dt", type=float, default=0.02, help="time-step size")
    ap.add_argument("--rho", type=float, default=1.0, help="fluid density")
    ap.add_argument("--gx", type=float, default=1.0e-4, help="streamwise body force")
    ap.add_argument("--nu0", type=float, default=0.1, help="zero-shear kinematic viscosity")
    ap.add_argument("--nu-inf", type=float, default=0.1,
                    help="infinite-shear viscosity (equal to nu0 => Newtonian)")
    ap.add_argument("--lambda", dest="lam", type=float, default=1.0, help="Carreau-Yasuda relaxation time")
    ap.add_argument("--n-cy", type=float, default=0.5, help="power-law index (<1 => shear thinning)")
    ap.add_argument("--a-cy", type=float, default=2.0, help="Yasuda transition exponent")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    # Explicit-diffusion stability check (2-D, both directions):
    #   dt <= h^2 / (4 * nu_max).  Violating it makes the explicit predictor blow
    #   up, so we assert here rather than letting the demo diverge silently.
    nu_max = max(args.nu0, args.nu_inf)
    dt_limit = args.h * args.h / (4.0 * nu_max)
    assert args.dt < dt_limit, (
        f"dt={args.dt} violates the diffusive stability limit "
        f"dt < h^2/(4*nu)={dt_limit:.4g}; reduce dt or nu."
    )

    vals = [args.nx, args.ny, args.steps, args.p_iters,
            args.h, args.dt, args.rho, args.gx,
            args.nu0, args.nu_inf, args.lam, args.n_cy, args.a_cy]
    line = " ".join(f"{v:g}" for v in vals)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(line + "\n", encoding="utf-8")
    mode = "Newtonian" if args.nu0 == args.nu_inf else "non-Newtonian (shear-thinning)"
    print(f"[make_synthetic] wrote {args.out}")
    print(f"[make_synthetic]   {args.nx}x{args.ny} grid, {args.steps} steps, "
          f"{mode}, dt={args.dt} (< stability limit {dt_limit:.4g}); SYNTHETIC")


if __name__ == "__main__":
    main()
