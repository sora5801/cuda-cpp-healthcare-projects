#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic phantom for 5.4
# ---------------------------------------------------------------------------
# Project 5.4 : Collapsed-Cone / Superposition-Convolution Dose  (2-D teaching model)
#
# WHY THIS EXISTS
#   Real dose-engine benchmark data (AAPM TG-105, IROC lung phantom, TCIA clinical
#   plans) either require registration or forbid redistribution, so we cannot
#   commit them. This script deterministically builds a TINY, clearly-SYNTHETIC
#   2-D density phantom that still exercises the whole algorithm -- including the
#   one feature that makes SC dose worth the trouble: HETEROGENEITY CORRECTION.
#
#   The phantom is a horizontal-slab stack (the beam enters the TOP, y=0, and
#   travels DOWN). Densities are water-relative (water = 1.0):
#       rows  0.. 3 : water   (rho = 1.0)   -- soft tissue build-up region
#       rows  4.. 7 : LUNG    (rho = 0.25)  -- low density: dose spreads FARTHER
#       rows  8..11 : water   (rho = 1.0)   -- soft tissue again
#       rows 12..13 : BONE    (rho = 1.85)  -- high density: dose spreads LESS
#       rows 14..15 : water   (rho = 1.0)
#   Watching the central-axis depth-dose curve, the learner SEES the kernel reach
#   deeper through the lung and pile up at the bone interface -- exactly the effect
#   a naive "scale by depth in water" model gets wrong and SC dose gets right.
#
#   Everything here is SYNTHETIC and for TEACHING ONLY -- not clinical data.
#
# OUTPUT FORMAT (whitespace-separated; see data/README.md):
#   line 1 : nx ny voxel_cm mu_over_rho psi0 n_cones kernel_a dose_scale beam_x0 beam_x1
#   then   : nx*ny density values, row-major (y=0 first), one row per text line.
#
# USAGE
#   python scripts/make_synthetic.py                 # writes data/sample/phantom.txt
#   python scripts/make_synthetic.py --out other.txt
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "phantom.txt"

# --- Geometry / physics constants (documented, deliberately small) ------------
NX, NY        = 16, 16          # 16 x 16 voxels: tiny, so the demo is instant + readable
VOXEL_CM      = 0.5             # 0.5 cm voxels -> an 8 x 8 cm field of view
MU_OVER_RHO   = 0.06            # cm^2/g: near water's mass-attenuation for ~1-2 MeV photons
PSI0          = 100.0           # incident primary fluence at the surface (arbitrary units)
N_CONES       = 8               # collapsed-cone directions (this model's max; see ccc_physics.h)
KERNEL_A      = 1.2             # cone kernel decay (1 / (g/cm^2)): dose-spread range
DOSE_SCALE    = 1.0e6           # fixed-point: 1 dose unit = 1e-6 dose (exact integer atomics)
BEAM_X0, BEAM_X1 = 6, 9         # a 4-voxel-wide central beam (columns 6..9 -> 2 cm wide)


def build_density():
    """Return the nx*ny row-major density map described in the header."""
    rho = [[1.0] * NX for _ in range(NY)]   # start everything as water
    for y in range(NY):
        if 4 <= y <= 7:
            band = 0.25     # lung
        elif 12 <= y <= 13:
            band = 1.85     # bone
        else:
            band = 1.0      # water
        for x in range(NX):
            rho[y][x] = band
    return rho


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic 5.4 phantom.")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    rho = build_density()
    header = (f"{NX} {NY} {VOXEL_CM} {MU_OVER_RHO} {PSI0} "
              f"{N_CONES} {KERNEL_A} {DOSE_SCALE:.0f} {BEAM_X0} {BEAM_X1}")
    body = "\n".join(" ".join(f"{v:g}" for v in row) for row in rho)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(header + "\n" + body + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  "
          f"({NX}x{NY} phantom: water/lung/water/bone/water; SYNTHETIC)")


if __name__ == "__main__":
    main()
