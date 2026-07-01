#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Write a synthetic adaptive-radiotherapy case
# ---------------------------------------------------------------------------
# Project 5.5 : Deformable Dose Accumulation & Adaptive Radiotherapy
#               (reduced-scope 2-D teaching version)
#
# WHAT THIS MAKES  (clearly SYNTHETIC -- no patient data anywhere in this repo)
#   One ART "case": four co-registered 2-D grids that make the deformable dose
#   accumulation result obvious and verifiable.
#
#     * PLANNING IMAGE  : a smooth bright BLOB (2-D Gaussian) centered in the
#                         frame -- stands in for the tumour on the planning CT.
#     * DAILY IMAGE     : the SAME blob, but SHIFTED by a known displacement and
#                         mildly stretched -- today's anatomy (the tumour moved).
#                         A good deformable registration must recover a spatially-
#                         varying vector field (not just one global translation).
#     * PLANNING DOSE   : a Gaussian dose "cloud" (Gy) centered on the PLANNING
#                         blob -- the dose the plan intends to deliver.
#     * DAILY DOSE      : the dose actually laid down TODAY. The linac fires the
#                         same beams, so the dose cloud is centered where the plan
#                         put it -- but the anatomy under it has MOVED. Warping the
#                         daily dose back by the DVF (deformable accumulation) puts
#                         the dose where it truly landed in the body; ignoring the
#                         motion (rigid accumulation) mis-places it. The demo shows
#                         the hot-spot difference between the two.
#
#   Both anatomy images are smooth Gaussians, so the intensity gradient is defined
#   everywhere -- exactly what Thirion's Demons force needs. Registering DAILY onto
#   PLANNING drives the SSD far down and recovers a ~4.5 px mean displacement that
#   matches the built-in shift+stretch, so the result is verifiable, not just
#   plausible.
#
# OUTPUT (data/README.md documents this exact format), whitespace-separated:
#   line 1 : "nx ny"
#   then   : nx*ny planning-image intensities in [0,1], row-major
#   then   : nx*ny daily-image    intensities in [0,1]
#   then   : nx*ny planning-dose  values in Gy
#   then   : nx*ny daily-dose     values in Gy
#
# USAGE
#   python scripts/make_synthetic.py
#   python scripts/make_synthetic.py --nx 96 --ny 96 --shift 4.0 --peak-dose 2.0
#
# Deterministic: no RNG, so the committed sample is reproducible byte-for-byte.
# ===========================================================================
import argparse
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "art_case.txt"


def gaussian(nx, ny, cx, cy, sx, sy, peak):
    """A smooth 2-D Gaussian of given peak, centered at (cx,cy) with per-axis
    std-devs (sx,sy). Returns a row-major list of nx*ny values. Smoothness -> a
    well-defined gradient everywhere (the Demons force relies on it) and a
    physically-plausible dose falloff."""
    img = [0.0] * (nx * ny)
    inv2sx2 = 1.0 / (2.0 * sx * sx)
    inv2sy2 = 1.0 / (2.0 * sy * sy)
    for y in range(ny):
        for x in range(nx):
            dx = x - cx
            dy = y - cy
            img[y * nx + x] = peak * math.exp(-(dx * dx) * inv2sx2
                                              - (dy * dy) * inv2sy2)
    return img


def main():
    ap = argparse.ArgumentParser(description="Write a synthetic ART case.")
    ap.add_argument("--nx", type=int, default=64, help="grid width in voxels")
    ap.add_argument("--ny", type=int, default=64, help="grid height in voxels")
    ap.add_argument("--sigma", type=float, default=9.0, help="blob width (voxels)")
    ap.add_argument("--shift", type=float, default=5.0,
                    help="tumour translation applied to the DAILY anatomy (voxels)")
    ap.add_argument("--stretch", type=float, default=0.12,
                    help="fractional non-uniform stretch of the daily blob")
    ap.add_argument("--dose-sigma", type=float, default=11.0,
                    help="dose-cloud width (voxels); a bit wider than the target")
    ap.add_argument("--peak-dose", type=float, default=2.0,
                    help="peak dose per fraction (Gy)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    nx, ny = args.nx, args.ny
    cx, cy = nx / 2.0, ny / 2.0

    # --- Anatomy images ---------------------------------------------------
    # PLANNING (fixed): a centered isotropic blob.
    plan_img = gaussian(nx, ny, cx, cy, args.sigma, args.sigma, 1.0)

    # DAILY (moving): the blob moved by +shift in x and +0.6*shift in y, and made
    # ~12% wider in x, so the true deformation is NOT a pure translation -> the DVF
    # must vary across the frame.
    mcx = cx + args.shift
    mcy = cy + 0.6 * args.shift
    msx = args.sigma * (1.0 + args.stretch)
    daily_img = gaussian(nx, ny, mcx, mcy, msx, args.sigma, 1.0)

    # --- Dose maps --------------------------------------------------------
    # PLANNING DOSE: a Gaussian dose cloud centered on the PLANNING target,
    # slightly wider than the target (typical PTV margin). Reported for context.
    plan_dose = gaussian(nx, ny, cx, cy, args.dose_sigma, args.dose_sigma,
                         args.peak_dose)

    # DAILY DOSE: the beams fire toward the PLANNED location, so the delivered
    # cloud is centered where the plan put it (same center as plan_dose). But the
    # tumour (daily_img) has moved away from that center -- so relative to the
    # tumour the dose is off-target. Warping the daily dose by the DVF (which maps
    # planning<-daily anatomy) re-expresses it in the planning frame; that is the
    # anatomically-correct accumulated dose. Here daily_dose == plan_dose on the
    # grid, which is the cleanest way to isolate the effect of the DVF.
    daily_dose = gaussian(nx, ny, cx, cy, args.dose_sigma, args.dose_sigma,
                          args.peak_dose)

    # Serialize. 6 decimals is plenty for [0,1] intensities and Gy doses, and keeps
    # the file small (the whole 64x64 case is ~110 KB of text, well under any limit).
    lines = [f"{nx} {ny}"]
    for grid in (plan_img, daily_img, plan_dose, daily_dose):
        lines.append(" ".join(f"{v:.6f}" for v in grid))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[make_synthetic] wrote {args.out}")
    print(f"  {nx}x{ny} ART case: planning blob at ({cx:.1f},{cy:.1f}), "
          f"daily blob at ({mcx:.1f},{mcy:.1f}), stretch={args.stretch}")
    print(f"  dose cloud sigma={args.dose_sigma} vox, peak={args.peak_dose} Gy/fraction")
    print("  synthetic data -- NOT patient-derived, NOT for clinical use.")


if __name__ == "__main__":
    main()
