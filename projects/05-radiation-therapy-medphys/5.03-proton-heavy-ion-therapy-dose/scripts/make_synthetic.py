#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic proton-plan sample
# ---------------------------------------------------------------------------
# Project 5.3 : Proton & Heavy-Ion Therapy Dose
#
# WHY THIS EXISTS
#   Real proton treatment plans are DICOM-RT (RTPLAN/RTDOSE) files derived from
#   patient CT scans; they are patient data and cannot be redistributed
#   (CLAUDE.md §8). So we ship a TINY, clearly-SYNTHETIC plan in the tiny text
#   format the loader (src/reference_cpu.cpp::load_plan) understands, so the demo
#   runs offline with zero downloads. Everything here is synthetic and labeled so.
#
#   The default plan written here is IDENTICAL to the built-in fallback in
#   src/main.cu (make_synthetic_plan): a SINGLE on-axis spot of range 12 cm, so
#   the central-axis depth-dose curve is the textbook PRISTINE BRAGG CURVE -- low
#   entrance plateau, sharp peak near 12 cm, hard zero beyond. Keeping the file
#   and the built-in fallback in lockstep means expected_output.txt is the same
#   whether you pass the file or run with no arguments.
#
#   Pass --ranges to stack several spots into a SPREAD-OUT BRAGG PEAK (SOBP), the
#   flat-topped plateau clinical plans use to cover a whole tumour (README
#   §Exercises). That produces a different (still valid) output, so it is offered
#   as an exercise, not the committed default.
#
# FILE FORMAT (whitespace-tolerant; '#' begins a comment line)
#   nx ny nz dx ox oy oz                         # grid: counts, spacing(cm), origin(cm)
#   sigma0 sigma_grow peak_width p_exp z_entry    # beam model + surface depth(cm)
#   n_spots                                       # number of spots
#   x0 y0 range weight        (repeated n_spots times; positions & range in cm)
#
# USAGE
#   python scripts/make_synthetic.py                        # writes the default sample
#   python scripts/make_synthetic.py --ranges 6 8 10 12     # custom SOBP peaks (cm)
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "proton_plan_sample.txt"

# The default beam model MUST match proton_physics.h::default_beam_model() and
# the grid + spots MUST match src/main.cu::make_synthetic_plan() so the file and
# the built-in fallback produce byte-identical output.
DEFAULT_GRID = (9, 9, 40, 0.5, -2.25, -2.25, 0.0)          # nx ny nz dx ox oy oz
DEFAULT_BEAM = (0.30, 0.020, 0.60, 1.77, 0.0)               # sigma0 grow peakw p_exp z_entry
# Default: a single 12 cm spot -> one pristine Bragg peak. (range_cm, weight).
DEFAULT_SPOTS = [(12.0, 1.00)]


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic proton-plan sample.")
    ap.add_argument("--ranges", type=float, nargs="+", default=None,
                    help="override on-axis spot ranges in cm; weights ramp 0.35..1.0 with depth")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    nx, ny, nz, dx, ox, oy, oz = DEFAULT_GRID
    sigma0, grow, peakw, p_exp, z_entry = DEFAULT_BEAM

    # Build the spot list. With --ranges, ramp weights linearly from 0.35 (proximal)
    # to 1.00 (distal) to approximate an SOBP; otherwise use the tuned defaults.
    if args.ranges:
        rs = sorted(args.ranges)
        n = len(rs)
        spots = [(R, round(0.35 + 0.65 * (i / max(1, n - 1)), 3)) for i, R in enumerate(rs)]
    else:
        spots = DEFAULT_SPOTS

    lines = []
    lines.append("# SYNTHETIC proton pencil-beam plan (teaching only; not clinical).")
    lines.append("# Project 5.3 -- Proton & Heavy-Ion Therapy Dose.")
    lines.append("# Default: one 12 cm spot = a pristine Bragg peak. --ranges stacks an SOBP.")
    lines.append("# grid: nx ny nz dx ox oy oz")
    lines.append(f"{nx} {ny} {nz} {dx:g} {ox:g} {oy:g} {oz:g}")
    lines.append("# beam: sigma0 sigma_grow peak_width p_exp z_entry")
    lines.append(f"{sigma0:g} {grow:g} {peakw:g} {p_exp:g} {z_entry:g}")
    lines.append("# spot count")
    lines.append(str(len(spots)))
    lines.append("# spots: x0 y0 range weight  (on axis; with --ranges, weight rises with depth)")
    for R, w in spots:
        lines.append(f"0 0 {R:g} {w:g}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    kind = "SOBP" if len(spots) > 1 else "pristine Bragg peak"
    print(f"[make_synthetic] wrote {args.out}  "
          f"(grid {nx}x{ny}x{nz}, {len(spots)} spot(s) [{kind}]; SYNTHETIC)")


if __name__ == "__main__":
    main()
