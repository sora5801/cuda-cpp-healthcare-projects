#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate a synthetic linac-QA dose plane pair
# ---------------------------------------------------------------------------
# Project 5.8 : Linac QA & Machine Performance Assessment  (catalog ID 5.8)
#
# WHAT THIS WRITES
#   A tiny, fully SYNTHETIC pair of 2-D dose planes for the gamma-index demo:
#     * the REFERENCE (planned) plane  -- an idealised open field, and
#     * the MEASURED  (EPID/portal) plane -- the same field with a small,
#       deterministic, physically-plausible machine error baked in.
#   No real patient or machine data is used or implied (CLAUDE.md §8).
#
# WHY THIS SHAPE
#   A real per-beam IMRT QA compares a measured portal-dosimetry image against
#   the planned fluence/dose. Here we model an OPEN FIELD (a flat top with a soft
#   penumbra) because its analytic form is exact and reproducible, and we inject
#   a realistic "healthy machine" error (1% low output + a mild right-side
#   asymmetry) so the gamma map is non-trivial yet mostly PASSING -- exactly what
#   a good daily QA looks like. This EXACTLY mirrors make_synthetic_qa() in
#   src/main.cu, so the committed file and the program's built-in fallback agree.
#
# FILE LAYOUT (whitespace-separated; parsed by src/reference_cpu.cpp load_qa):
#   nx ny spacing_mm dd_percent dta_mm norm_dose
#   <nx*ny reference-plane values, row-major>
#   <nx*ny measured-plane  values, row-major>
#
# USAGE
#   python scripts/make_synthetic.py
#   python scripts/make_synthetic.py --nx 24 --ny 24 --spacing 2.0
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "qa_planes_sample.txt"


def edge(pos: float, c: float, half_field: float, penumbra: float) -> float:
    """1 inside the flat top, ramping linearly to 0 across the penumbra.

    Mirrors the lambda in src/main.cu make_synthetic_qa(); |pos-c| is the
    distance from the plane centre along one axis (separable in x and y)."""
    d = abs(pos - c)
    if d <= half_field:
        return 1.0
    if d >= half_field + penumbra:
        return 0.0
    return 1.0 - (d - half_field) / penumbra


def build(nx: int, ny: int, half_field: float, penumbra: float):
    """Return (ref, meas) as flat row-major lists of length nx*ny."""
    cx = 0.5 * (nx - 1)   # plane centre (columns)
    cy = 0.5 * (ny - 1)   # plane centre (rows)
    ref, meas = [], []
    for y in range(ny):
        for x in range(nx):
            # Separable open field: flat top of value 100 with soft edges.
            prof = 100.0 * edge(x, cx, half_field, penumbra) \
                         * edge(y, cy, half_field, penumbra)
            ref.append(prof)
            # Measured = planned with a small realistic machine error:
            #   * 1% low output overall (a common daily drift), and
            #   * the right half 2% hotter (a mild left/right asymmetry).
            m = prof * 0.99
            if x > cx:
                m *= 1.02
            meas.append(m)
    return ref, meas


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a synthetic linac-QA dose-plane pair.")
    ap.add_argument("--nx", type=int, default=24, help="plane width in pixels")
    ap.add_argument("--ny", type=int, default=24, help="plane height in pixels")
    ap.add_argument("--spacing", type=float, default=2.0, help="pixel size, mm")
    ap.add_argument("--dd", type=float, default=3.0, help="dose-difference criterion, %%")
    ap.add_argument("--dta", type=float, default=3.0, help="distance-to-agreement, mm")
    ap.add_argument("--half-field", type=float, default=8.0, help="flat-top half-width, px")
    ap.add_argument("--penumbra", type=float, default=2.0, help="edge softness, px")
    args = ap.parse_args()

    ref, meas = build(args.nx, args.ny, args.half_field, args.penumbra)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    # NOTE: the loader (util::read_floats) does a bare `stream >> float`, which
    # stops at the first non-numeric token. So the file must contain ONLY numbers
    # -- no '#' comment lines. Provenance/labelling lives in data/README.md.
    with OUT.open("w", encoding="utf-8") as f:
        # Header: norm_dose = 0 tells the loader to normalise to the ref max.
        f.write(f"{args.nx} {args.ny} {args.spacing} {args.dd} {args.dta} 0\n")
        # Reference plane first (row-major), then the measured plane.
        for plane in (ref, meas):
            for y in range(args.ny):
                row = plane[y * args.nx:(y + 1) * args.nx]
                f.write(" ".join(f"{v:.4f}" for v in row) + "\n")
    print(f"wrote {OUT}  ({args.nx}x{args.ny} SYNTHETIC QA dose planes)")


if __name__ == "__main__":
    main()
