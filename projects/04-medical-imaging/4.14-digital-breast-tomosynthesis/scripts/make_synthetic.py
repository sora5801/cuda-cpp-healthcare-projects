#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic DBT projection sample
# ---------------------------------------------------------------------------
# Project 4.14 : Digital Breast Tomosynthesis
#
# WHY THIS EXISTS
#   Real DBT projection data (OPTIMAM, VinDr-Mammo, BCS-DBT) is credentialed
#   and/or non-redistributable (see data/README.md). So the committed demo runs
#   on a CLEARLY-SYNTHETIC stand-in generated here: a known compressed-breast
#   phantom, forward-projected over the SAME narrow angular wedge the C++ SART
#   reconstructs. Because we know the ground-truth phantom, the reconstruction is
#   verifiable by eye (the two planted lesions must reappear). Synthetic data is
#   LABELED synthetic everywhere (CLAUDE.md §8) -- this is NOT clinical data.
#
# THE PHANTOM (world square [-W, W]^2, attenuation in arbitrary units)
#   * a soft fibroglandular ELLIPSE (low attenuation) = the breast tissue, and
#   * two small dense DISCS (high attenuation) = simulated lesions/calcifications
#     at two known positions on the central row.
#   The phantom is rendered on a fine grid, then forward-projected analytically
#   (a numeric line integral matching the C++ forward model: sample the phantom
#   along each ray and sum * step length). Feeding these projections to SART must
#   recover elevated attenuation at the lesion sites.
#
# OUTPUT FORMAT (matches reference_cpu.cpp::load_dbt)
#   header: n_angles n_det ds img world_half half_span relax n_iters
#   then n_angles rows of n_det projection values.
#
# USAGE
#   python scripts/make_synthetic.py                 # writes data/sample/dbt_sample.txt
#   python scripts/make_synthetic.py --img 128 --angles 21   # bigger problem
# ===========================================================================
import argparse
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "dbt_sample.txt"


def phantom_value(x, y, W):
    """Ground-truth attenuation at world point (x, y).

    Soft elliptical breast tissue + two dense lesion discs. Units are arbitrary
    (this is a synthetic software phantom, not calibrated Hounsfield/mu values).
    """
    mu = 0.0
    # Fibroglandular ellipse: semi-axes 0.62W (x) by 0.42W (y), centred at origin.
    if (x / (0.62 * W)) ** 2 + (y / (0.42 * W)) ** 2 <= 1.0:
        mu += 0.20                                     # soft-tissue background
    # Lesion disc 1: left of centre, on the central row (y = 0).
    if (x + 0.28 * W) ** 2 + y ** 2 <= (0.09 * W) ** 2:
        mu += 0.80                                     # dense lesion
    # Lesion disc 2: right of centre, on the central row (y = 0).
    if (x - 0.30 * W) ** 2 + y ** 2 <= (0.06 * W) ** 2:
        mu += 0.60                                     # smaller dense lesion
    return mu


def forward_project(n_angles, n_det, ds, W, half_span, steps):
    """Numeric line integrals of the phantom over the narrow DBT wedge.

    Mirrors the C++ forward model (dbt_geometry.h::forward_ray_integral):
    for each angle theta_k and detector bin j, march along the ray
      p(t) = s*(cos,sin) + t*(-sin,cos),  t in [-L, L],  L = sqrt(2)*W
    sampling the phantom and summing * per-step length. This keeps the synthetic
    data consistent with what the reconstructor expects.
    """
    proj = []
    center = 0.5 * (n_det - 1)
    L = math.sqrt(2.0) * W
    dt = (2.0 * L / (steps - 1)) if steps > 1 else 0.0
    angle_step = (2.0 * half_span / (n_angles - 1)) if n_angles > 1 else 0.0
    for k in range(n_angles):
        theta = -half_span + k * angle_step
        ck, sk = math.cos(theta), math.sin(theta)
        row = []
        for j in range(n_det):
            s = (j - center) * ds
            acc = 0.0
            for m in range(steps):
                t = -L + m * dt
                x = s * ck - t * sk
                y = s * sk + t * ck
                acc += phantom_value(x, y, W)
            row.append(acc * dt)                       # numeric line integral
        proj.append(row)
    return proj


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic DBT sample.")
    ap.add_argument("--angles", type=int, default=15, help="projection count (DBT: 9-25)")
    ap.add_argument("--det", type=int, default=96, help="detector bins")
    ap.add_argument("--img", type=int, default=64, help="reconstruction side length")
    ap.add_argument("--world-half", type=float, default=1.0, help="image half-extent W")
    ap.add_argument("--span-deg", type=float, default=25.0, help="HALF angular span (deg)")
    ap.add_argument("--relax", type=float, default=0.20, help="SART relaxation lambda")
    ap.add_argument("--iters", type=int, default=12, help="SART iterations")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    W = args.world_half
    half_span = math.radians(args.span_deg)
    # Detector spacing: cover a bit more than the image diagonal so no ray of the
    # in-image phantom falls off the detector.
    ds = (2.2 * W) / (args.det - 1)
    steps = 2 * args.img                               # matches n_ray_steps() in C++

    proj = forward_project(args.angles, args.det, ds, W, half_span, steps)

    lines = [f"{args.angles} {args.det} {ds:.9g} {args.img} {W:.9g} "
             f"{half_span:.9g} {args.relax:.9g} {args.iters}"]
    for row in proj:
        lines.append(" ".join(f"{v:.6f}" for v in row))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}")
    print(f"[make_synthetic]   {args.angles} angles over +/-{args.span_deg} deg, "
          f"{args.det} detectors, {args.img}x{args.img} image; SYNTHETIC phantom "
          f"(2 lesions). NOT clinical data.")


if __name__ == "__main__":
    main()
