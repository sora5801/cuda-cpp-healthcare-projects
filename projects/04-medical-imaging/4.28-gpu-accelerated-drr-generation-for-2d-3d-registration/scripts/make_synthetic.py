#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic CT volume sample
# ---------------------------------------------------------------------------
# Project 4.28 : GPU-Accelerated DRR Generation for 2D/3D Registration
#
# WHY THIS EXISTS
#   Real CT volumes (TCIA, Gold Atlas, clinical CBCT) are large and/or
#   credential-gated, and several cannot be redistributed. So the committed demo
#   runs on a TINY, fully SYNTHETIC CT phantom generated here. It is labeled
#   synthetic everywhere (data/README.md) and carries NO clinical meaning.
#
# THE PHANTOM (engineered so the DRR is interpretable, PATTERNS.md section 6)
#   A small cubic grid of Hounsfield Units containing three nested structures:
#     * AIR background         : HU = -1000  (mu ~ 0)         -> transparent
#     * SOFT-TISSUE sphere     : HU =   +40  (~ muscle/water) -> faint shadow
#     * DENSE BONE sphere      : HU = +1000  (cortical bone)  -> brightest spot,
#                                deliberately OFFSET from center so the brightest
#                                DRR pixel lands off-axis -- a result the learner
#                                can predict from the geometry and verify in the
#                                program's "max attenuation at (u,v)" line.
#   Because the bone ball is offset, the DRR is asymmetric: a good visual + numeric
#   check that the ray geometry is wired up correctly.
#
# FILE FORMAT (read by src/reference_cpu.cpp::load_volume)
#     nx ny nz sx sy sz          # dims (ints) then voxel spacing in mm (floats)
#     hu hu hu ... (nx*ny*nz)    # Hounsfield Units, row-major [z][y][x], x fastest
#
# DETERMINISM: no randomness at all -- the phantom is a pure function of (nx,ny,nz)
# and the spacing, so re-running this reproduces byte-identical data, and the
# demo's expected_output.txt stays stable.
#
# USAGE
#   python scripts/make_synthetic.py                 # default 32^3 committed sample
#   python scripts/make_synthetic.py --n 64          # bigger volume (not committed)
#   python scripts/make_synthetic.py --out other.txt
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent           # the project folder
OUT = ROOT / "data" / "sample" / "ct_volume_sample.txt"

# Hounsfield Unit constants for the three materials (textbook diagnostic values).
HU_AIR  = -1000.0
HU_SOFT =    40.0
HU_BONE =  1000.0


def build_phantom(nx, ny, nz):
    """Return a flat list of HU values (row-major [z][y][x]) for the phantom.

    All coordinates are in voxel units; the spheres are placed as fractions of the
    grid so the phantom scales sensibly with `n`. The bone sphere is intentionally
    offset toward +x/+y so the projection is asymmetric.
    """
    # Centers (voxel coords) and radii (voxels) of the two spheres.
    cx, cy, cz = (nx - 1) / 2.0, (ny - 1) / 2.0, (nz - 1) / 2.0
    soft_r = 0.42 * nx                                   # large, fills most of the volume
    # Bone sphere: offset from center, smaller -> a compact dense target.
    bx, by, bz = cx + 0.18 * nx, cy + 0.12 * ny, cz
    bone_r = 0.14 * nx

    data = []
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                # Squared distances to each sphere center (avoid sqrt: compare r^2).
                ds = (ix - cx) ** 2 + (iy - cy) ** 2 + (iz - cz) ** 2
                db = (ix - bx) ** 2 + (iy - by) ** 2 + (iz - bz) ** 2
                if db <= bone_r ** 2:
                    hu = HU_BONE                          # dense bone wins (it is inside soft)
                elif ds <= soft_r ** 2:
                    hu = HU_SOFT                          # soft-tissue body
                else:
                    hu = HU_AIR                           # surrounding air
                data.append(hu)
    return data


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic CT phantom for DRR.")
    ap.add_argument("--n", type=int, default=32, help="cubic volume side (voxels)")
    ap.add_argument("--spacing", type=float, default=2.0, help="voxel spacing in mm")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    n = args.n
    sp = args.spacing
    data = build_phantom(n, n, n)

    # Header + body. Integers print without a trailing ".0" via :g for compactness.
    header = f"{n} {n} {n} {sp:g} {sp:g} {sp:g}"
    # One z-slice per line keeps the file human-skimmable while staying valid
    # (the loader reads purely by whitespace, so line breaks are cosmetic).
    body_lines = []
    for iz in range(n):
        start = iz * n * n
        row = data[start:start + n * n]
        body_lines.append(" ".join(f"{v:g}" for v in row))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(header + "\n" + "\n".join(body_lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {out_path}  "
          f"({n}x{n}x{n} voxels, spacing {sp} mm; SYNTHETIC phantom: air+soft+offset-bone)")


if __name__ == "__main__":
    main()
