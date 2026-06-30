#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic CT-like volume
# ---------------------------------------------------------------------------
# Project 4.18 : Image-Based 3D Printing / Model Generation for Surgery
#
# WHY THIS EXISTS
#   Real clinical CT volumes (TCIA, the OsteoArthritis Initiative, the
#   TotalSegmentator dataset) require registration/credentials and cannot be
#   redistributed wholesale (see data/README.md and CLAUDE.md sec.8). So that the
#   demo runs OFFLINE with zero downloads, this script deterministically
#   generates a tiny, clearly-SYNTHETIC scalar volume whose isosurface is a known
#   shape -- a SPHERE -- so the extracted mesh has a verifiable analytic area.
#
# WHAT IT WRITES  (the format src/reference_cpu.cpp::load_problem reads)
#   line 1 : nx ny nz spacing origin_x origin_y origin_z iso
#   rest   : nx*ny*nz floats, row-major with x fastest, then y, then z.
#
#   The field is a smooth "implicit sphere":
#       value(i,j,k) = radius - distance_from_center(i,j,k)   (in mm)
#   so value == 0 is exactly the sphere surface of the given radius. We extract
#   at iso = 0.0. Because the field is signed distance, marching cubes places the
#   surface essentially on the true sphere, and 4*pi*r^2 is the analytic area the
#   demo cross-checks against (see src/main.cu [science] line).
#
#   Everything here is SYNTHETIC and labeled as such; it is NOT patient data and
#   carries no clinical meaning.
#
# USAGE
#   python scripts/make_synthetic.py                 # default 17^3 sphere
#   python scripts/make_synthetic.py --n 33 --radius 12
# ===========================================================================
import argparse
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "volume_sample.txt"


def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic sphere volume for marching cubes.")
    ap.add_argument("--n", type=int, default=17,
                    help="samples per axis (cube volume n x n x n); keep small for a committed sample")
    ap.add_argument("--spacing", type=float, default=1.0, help="mm between samples")
    ap.add_argument("--radius", type=float, default=6.0, help="sphere radius in mm")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    n, spacing, radius = args.n, args.spacing, args.radius

    # Center the sphere in the middle of the grid (in sample-index space). The
    # origin is chosen so the grid is centered on (0,0,0) in world coords, which
    # keeps the printed bounding box symmetric and easy to read.
    center = (n - 1) / 2.0
    origin = -center * spacing  # world coord of sample index 0 along each axis
    iso = 0.0                   # extract the value==0 surface = the true sphere

    # Build the flat sample array in (k, j, i) order -- x fastest -- matching the
    # loader. value = radius - |p - c| is a signed-distance field: positive
    # (inside, "denser than threshold") within the sphere, negative outside.
    vals = []
    for k in range(n):
        for j in range(n):
            for i in range(n):
                # World-space offset of this sample from the sphere center (mm).
                dx = (i - center) * spacing
                dy = (j - center) * spacing
                dz = (k - center) * spacing
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                vals.append(radius - dist)

    header = f"{n} {n} {n} {spacing:g} {origin:g} {origin:g} {origin:g} {iso:g}"
    # Group values 8 per line for readability; the loader ignores line breaks.
    body_lines = []
    for off in range(0, len(vals), 8):
        body_lines.append(" ".join(f"{v:.4f}" for v in vals[off:off + 8]))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(header + "\n" + "\n".join(body_lines) + "\n",
                              encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  "
          f"({n}^3 sphere, r={radius} mm, spacing={spacing} mm; SYNTHETIC)")


if __name__ == "__main__":
    main()
