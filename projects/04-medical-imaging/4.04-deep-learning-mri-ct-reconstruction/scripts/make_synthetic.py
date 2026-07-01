#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic MRI acquisition sample
# ---------------------------------------------------------------------------
# Project 4.4 : Deep-Learning MRI/CT Reconstruction  (REDUCED-SCOPE TEACHING VERSION)
#
# WHY THIS EXISTS
#   The real fastMRI raw multi-coil k-space cannot be redistributed (it requires
#   registration + a license; see data/README.md and scripts/download_data.*).
#   So the committed demo runs on a CLEARLY-SYNTHETIC stand-in generated here: a
#   tiny piecewise-constant phantom, its 2-D DFT, and an under-sampling mask --
#   the SAME construction the C++ make_synthetic_acquisition() uses, so the file
#   and the built-in fallback describe the same scan.
#
#   Everything this writes is SYNTHETIC and labeled as such. No patient data.
#
# FILE LAYOUT (whitespace-separated floats; parsed by load_acquisition):
#     ny nx
#     truth[0 .. N-1]            (N = ny*nx, row-major)
#     mask[0 .. N-1]             (0/1)
#     kmeas_re[0 .. N-1]
#     kmeas_im[0 .. N-1]
#
#   We compute the DFT in float32 to stay close to the C++ float path. The demo's
#   expected_output.txt is captured from a REAL run on THIS file, so the exact
#   bit pattern of the DFT does not need to match the C++ one -- only be fixed.
#
# USAGE
#   python scripts/make_synthetic.py                 # writes the 24x24 sample
#   python scripts/make_synthetic.py --ny 32 --nx 32 # a bigger phantom
# ===========================================================================
import argparse
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "mri_scan_sample.txt"


def build_truth(ny, nx):
    """Piecewise-constant phantom: dim background + bright disk + gray square.
    Mirrors make_synthetic_acquisition() in src/reference_cpu.cpp."""
    truth = [0.1] * (ny * nx)                           # dim background
    cy, cx = (ny - 1) * 0.5, (nx - 1) * 0.5             # image center
    r = 0.30 * nx                                       # disk radius (pixels)
    for y in range(ny):
        for x in range(nx):
            if (y - cy) ** 2 + (x - cx) ** 2 <= r * r:
                truth[y * nx + x] = 1.0                 # bright disk
            if ny // 6 <= y < ny // 6 + ny // 4 and nx // 6 <= x < nx // 6 + nx // 4:
                truth[y * nx + x] = 0.6                 # mid-gray square
    return truth


def forward_dft(truth, ny, nx):
    """Direct (un-shifted) 2-D DFT, matching dft_core.h's convention:
       F[v,u] = sum_{y,x} img[y,x] * exp(-2pi i (v*y/ny + u*x/nx)).
    O(N^2) but N is tiny. Returns (re, im) as flat row-major lists."""
    re = [0.0] * (ny * nx)
    im = [0.0] * (ny * nx)
    two_pi = 2.0 * math.pi
    for v in range(ny):
        for u in range(nx):
            sr = si = 0.0
            for y in range(ny):
                py = two_pi * (v * y / ny)
                for x in range(nx):
                    ang = py + two_pi * (u * x / nx)
                    p = truth[y * nx + x]
                    sr += p * math.cos(ang)
                    si -= p * math.sin(ang)            # forward sign
            re[v * nx + u] = sr
            im[v * nx + u] = si
    return re, im


def build_mask(ny, nx):
    """Under-sampling mask: narrow low-frequency band (always kept) + every 2nd
    line elsewhere. Mirrors reference_cpu.cpp (lowband = ny//8)."""
    mask = [0] * (ny * nx)
    lowband = ny // 8
    for v in range(ny):
        v_low = (v < lowband) or (v >= ny - lowband)
        v_keep = v_low or (v % 2 == 0)
        for u in range(nx):
            u_low = (u < lowband) or (u >= nx - lowband)
            if v_keep and (u_low or (u % 2 == 0)):
                mask[v * nx + u] = 1
    return mask


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic MRI acquisition sample.")
    ap.add_argument("--ny", type=int, default=24, help="image height (pixels)")
    ap.add_argument("--nx", type=int, default=24, help="image width (pixels)")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()
    ny, nx = args.ny, args.nx

    truth = build_truth(ny, nx)
    fre, fim = forward_dft(truth, ny, nx)
    mask = build_mask(ny, nx)

    # Zero the unsampled bins so kmeas is exactly what the scanner "measured".
    kre = [fre[i] if mask[i] else 0.0 for i in range(ny * nx)]
    kim = [fim[i] if mask[i] else 0.0 for i in range(ny * nx)]

    def row(vals):
        return " ".join(f"{v:.8g}" for v in vals)

    lines = [f"{ny} {nx}", row(truth), row(mask), row(kre), row(kim)]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    kept = sum(mask)
    print(f"[make_synthetic] wrote {args.out}  ({ny}x{nx}, {kept}/{ny*nx} bins kept; SYNTHETIC)")


if __name__ == "__main__":
    main()
