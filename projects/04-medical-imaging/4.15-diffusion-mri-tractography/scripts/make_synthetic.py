#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate a synthetic DWI volume with a known
#                                 fiber bundle (for DTI fit + tractography)
# ---------------------------------------------------------------------------
# Project 4.15 : Diffusion MRI & Tractography
#
# WHY SYNTHETIC
#   Real diffusion MRI comes from HCP / UK Biobank / ABCD (see download_data.*
#   and data/README.md), which require registration and are far too large to
#   commit. To keep the demo offline, reproducible, and INTERPRETABLE we forward-
#   simulate a tiny volume from the Stejskal-Tanner signal equation with a KNOWN
#   ground-truth tensor field: a curved anisotropic fiber bundle embedded in an
#   isotropic background. Fitting should recover high FA along the bundle and the
#   correct fiber orientation, and tractography should reconstruct the curve.
#
# THE FORWARD MODEL (the inverse of what src/ solves)
#   For each voxel we choose a diffusion tensor D (its eigenvectors/eigenvalues),
#   then for each measurement k compute  S_k = S0 * exp(-b_k * g_k^T D g_k).
#   Voxels ON the bundle get an anisotropic (cigar-shaped) tensor pointing along
#   the local bundle tangent; background voxels get an isotropic tensor (a
#   sphere, FA~0). NO NOISE is added, so the fit is exact and the demo output is
#   perfectly deterministic (see THEORY "Numerical considerations").
#
# GRADIENT SCHEME
#   Must match src/reference_cpu.cpp::make_gradient_scheme exactly: 1 b=0 image +
#   12 icosahedral directions at b = 1000 s/mm^2.
#
# OUTPUT FORMAT (data/README.md):
#   line 1 : "<nx> <ny> <nz> <nmeas>"
#   then, per voxel (x fastest, then y, then z):  <mask> S_0 S_1 ... S_{nmeas-1}
#
# USAGE
#   python scripts/make_synthetic.py                 # default 16x16x4 volume
#   python scripts/make_synthetic.py --nx 64 --ny 64 --nz 32   # a bigger set
# ===========================================================================
import argparse
import math
from pathlib import Path

# MUST match src/dti_core.h.
NDIR = 12
NMEAS = 1 + NDIR
BVAL = 1000.0            # s/mm^2 (single shell)
S0 = 1000.0             # baseline (b=0) signal intensity

# Eigenvalues (mm^2/s) for the two tissue types.
LAM_PAR = 1.7e-3        # diffusivity ALONG a fiber (fast)
LAM_PERP = 0.3e-3       # diffusivity ACROSS a fiber (slow) -> high FA
LAM_ISO = 0.9e-3        # isotropic background (FA ~ 0)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "dwi_sample.txt"


def gradient_scheme():
    """The 1 + 12 icosahedral scheme, identical to reference_cpu.cpp."""
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    inv = 1.0 / math.sqrt(1.0 + phi * phi)
    raw = [
        (0, 1, phi), (0, 1, -phi), (0, -1, phi), (0, -1, -phi),
        (1, phi, 0), (1, -phi, 0), (-1, phi, 0), (-1, -phi, 0),
        (phi, 0, 1), (-phi, 0, 1), (phi, 0, -1), (-phi, 0, -1),
    ]
    bval = [0.0] + [BVAL] * NDIR
    gx = [0.0] + [r[0] * inv for r in raw]
    gy = [0.0] + [r[1] * inv for r in raw]
    gz = [0.0] + [r[2] * inv for r in raw]
    return bval, gx, gy, gz


def tensor_from_dir(dx, dy, dz, lam_par, lam_perp):
    """Build a symmetric tensor D = lam_perp*I + (lam_par-lam_perp)*(d d^T),
    i.e. an axially-symmetric ('cigar') tensor with fast axis along unit d.
    Returns (Dxx,Dyy,Dzz,Dxy,Dxz,Dyz)."""
    n = math.sqrt(dx * dx + dy * dy + dz * dz)
    dx, dy, dz = dx / n, dy / n, dz / n
    a = lam_par - lam_perp
    Dxx = lam_perp + a * dx * dx
    Dyy = lam_perp + a * dy * dy
    Dzz = lam_perp + a * dz * dz
    Dxy = a * dx * dy
    Dxz = a * dx * dz
    Dyz = a * dy * dz
    return Dxx, Dyy, Dzz, Dxy, Dxz, Dyz


def signal(D, bval, gx, gy, gz):
    """Stejskal-Tanner: S_k = S0 * exp(-b_k * g_k^T D g_k)."""
    Dxx, Dyy, Dzz, Dxy, Dxz, Dyz = D
    out = []
    for k in range(NMEAS):
        gxx, gyy, gzz = gx[k], gy[k], gz[k]
        qDq = (gxx * gxx * Dxx + gyy * gyy * Dyy + gzz * gzz * Dzz
               + 2 * gxx * gyy * Dxy + 2 * gxx * gzz * Dxz + 2 * gyy * gzz * Dyz)
        out.append(S0 * math.exp(-bval[k] * qDq))
    return out


def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic DWI volume with a curved fiber bundle.")
    ap.add_argument("--nx", type=int, default=16)
    ap.add_argument("--ny", type=int, default=16)
    ap.add_argument("--nz", type=int, default=4)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    nx, ny, nz = args.nx, args.ny, args.nz

    bval, gx, gy, gz = gradient_scheme()

    # The ground-truth bundle: a quarter-circle arc in the x-y plane (repeated in
    # every z slice), centred at (0, ny-1). A voxel is "on the bundle" if it lies
    # within `width` of the arc of radius R; its fiber tangent is perpendicular to
    # the radius. This gives a CURVED tract that tractography must follow.
    R = 0.7 * min(nx, ny)          # arc radius
    width = 1.2                    # half-thickness of the bundle (voxels)
    cx, cy = 0.0, (ny - 1)         # arc centre

    lines = [f"{nx} {ny} {nz} {NMEAS}"]
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                r = math.hypot(x - cx, y - cy)
                on_bundle = abs(r - R) <= width
                if on_bundle and r > 1e-6:
                    # Tangent to the circle = perpendicular to the radius vector.
                    rxn, ryn = (x - cx) / r, (y - cy) / r
                    tx, ty, tz = -ryn, rxn, 0.0   # rotate radius by 90 deg in-plane
                    D = tensor_from_dir(tx, ty, tz, LAM_PAR, LAM_PERP)
                    mask = 1
                else:
                    # Isotropic background (a sphere): FA ~ 0.
                    D = (LAM_ISO, LAM_ISO, LAM_ISO, 0.0, 0.0, 0.0)
                    mask = 1 if on_bundle else 0
                S = signal(D, bval, gx, gy, gz)
                vals = " ".join(f"{s:.6f}" for s in S)
                lines.append(f"{mask} {vals}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  "
          f"({nx}x{ny}x{nz}={nx*ny*nz} voxels, {NMEAS} meas; SYNTHETIC curved bundle, no noise)")


if __name__ == "__main__":
    main()
