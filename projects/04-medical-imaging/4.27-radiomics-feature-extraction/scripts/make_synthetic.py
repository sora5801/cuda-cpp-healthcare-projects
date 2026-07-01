#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic radiomics ROI sample
# ---------------------------------------------------------------------------
# Project 4.27 : Radiomics Feature Extraction
#
# WHY THIS EXISTS
#   Real radiomics datasets (TCIA NSCLC-Radiomics, QIN-HEADNECK, ...) are large
#   DICOM CT/PET/MRI volumes with segmentation masks; some require registration
#   and none should be redistributed here. So the demo runs on a TINY, clearly
#   SYNTHETIC volume this script generates -- deterministically, so the committed
#   sample and the expected demo output never drift.
#
# WHAT WE GENERATE  (engineered to make the features INTERPRETABLE)
#   A small nx*ny*nz grid of "intensities" (arbitrary units, think Hounsfield)
#   with a spherical region-of-interest mask. INSIDE the ROI we lay down a
#   deterministic TEXTURE: a smooth radial gradient PLUS a coarse checkerboard
#   ripple. The gradient gives the GLCM a correlated (near-diagonal) component;
#   the checkerboard injects high-contrast neighbour pairs -- so contrast,
#   homogeneity and correlation all take non-trivial, reproducible values.
#   Voxels OUTSIDE the sphere are background (mask 0) and never contribute.
#
# OUTPUT FORMAT  (see data/README.md; parsed by load_volume in reference_cpu.cpp)
#   line 1 : "nx ny nz Ng"
#   then   : nx*ny*nz intensities  (row-major, x fastest, then y, then z)
#   then   : nx*ny*nz mask flags   (0/1, same ordering)
#
# USAGE
#   python scripts/make_synthetic.py                 # default 6x6x5, Ng=8
#   python scripts/make_synthetic.py --nx 16 --ny 16 --nz 12 --ng 16
# ===========================================================================
import argparse
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "radiomics_sample.txt"


def generate(nx, ny, nz, ng):
    """Return (intensities, mask) as flat row-major lists (x fastest)."""
    cx, cy, cz = (nx - 1) / 2.0, (ny - 1) / 2.0, (nz - 1) / 2.0
    # ROI radius: a sphere that fills most of the grid but leaves a background rim.
    radius = 0.5 * min(nx, ny, nz) - 0.001

    intensity = []
    mask = []
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                dx, dy, dz = x - cx, y - cy, z - cz
                r = math.sqrt(dx * dx + dy * dy + dz * dz)
                inside = 1 if r <= radius else 0
                mask.append(inside)
                if inside:
                    # Smooth radial gradient in [0, 60]: correlated texture.
                    grad = 60.0 * (1.0 - r / (radius + 1e-9))
                    # Coarse checkerboard ripple in {-20, +20}: high-contrast pairs.
                    ripple = 20.0 if ((x + y + z) % 2 == 0) else -20.0
                    intensity.append(round(40.0 + grad + ripple, 4))
                else:
                    # Background intensity (ignored by the mask, but present in
                    # the dense grid the loader reads).
                    intensity.append(0.0)
    return intensity, mask


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic radiomics ROI sample.")
    ap.add_argument("--nx", type=int, default=6, help="grid size x")
    ap.add_argument("--ny", type=int, default=6, help="grid size y")
    ap.add_argument("--nz", type=int, default=5, help="grid size z")
    ap.add_argument("--ng", type=int, default=8, help="number of gray levels (2..16)")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    intensity, mask = generate(args.nx, args.ny, args.nz, args.ng)

    lines = [f"{args.nx} {args.ny} {args.nz} {args.ng}"]
    # Intensities and mask, wrapped to nx values per line for readability.
    row = args.nx
    lines += [" ".join(f"{v:g}" for v in intensity[i:i + row])
              for i in range(0, len(intensity), row)]
    lines += [" ".join(str(m) for m in mask[i:i + row])
              for i in range(0, len(mask), row)]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    nroi = sum(mask)
    print(f"[make_synthetic] wrote {args.out}  "
          f"({args.nx}x{args.ny}x{args.nz}, Ng={args.ng}, {nroi} ROI voxels; SYNTHETIC)")


if __name__ == "__main__":
    main()
