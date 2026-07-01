#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic dose-comparison sample
# ---------------------------------------------------------------------------
# Project 5.9 -- Gamma-Index Dose Comparison
#
# WHY THIS EXISTS
#   Real IMRT/VMAT QA dose pairs come from clinical planning systems + measured
#   detector arrays (see data/README.md) and are NOT redistributable. So the
#   committed sample is fully SYNTHETIC: a smooth "dose hill" reference and an
#   evaluated map that differs from it in two controlled, gamma-predictable ways.
#   Synthetic data is always LABELED synthetic (CLAUDE.md §8).
#
#   The generated file EXACTLY mirrors what the C++ built-in make_synthetic()
#   produces (same 32x32 grid, same Gaussian, same +1.5% bias, same central hot
#   spot), so `demo/run_demo` on this file yields the same result the binary
#   produces with no argument -- and both are stable, giving a deterministic
#   expected_output.txt.
#
# FILE LAYOUT (whitespace-separated floats; parsed by src/main.cu load_sample):
#     width height
#     spacing_mm  dd_percent  dta_mm  dose_threshold_frac
#     <width*height reference dose values, row-major>
#     <width*height evaluated dose values, row-major>
#
# USAGE
#   python scripts/make_synthetic.py                 # writes the 32x32 sample
#   python scripts/make_synthetic.py --n 64          # a larger 64x64 problem
# ===========================================================================
import argparse
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent            # the project folder
OUT = ROOT / "data" / "sample" / "dose_pair.txt"


def build(n: int):
    """Return (width, height, params, ref, eval) for an n x n synthetic pair.

    Mirrors src/main.cu make_synthetic() so the committed sample and the C++
    built-in fallback produce identical numbers.
    """
    W = H = n
    spacing_mm = 2.0            # 2 mm voxels (typical QA array pitch)
    dd_percent = 3.0           # 3% / 3 mm -- the classic clinical criterion
    dta_mm = 3.0
    thresh_frac = 0.10         # analyze points above 10% of the max dose

    cx = 0.5 * (W - 1)
    cy = 0.5 * (H - 1)
    sigma = 8.0 * (n / 32.0)   # scale the hill width with the grid

    ref = []
    ev = []
    for y in range(H):
        for x in range(W):
            dx = x - cx
            dy = y - cy
            r2 = dx * dx + dy * dy
            dose = 100.0 * math.exp(-r2 / (2.0 * sigma * sigma))  # reference hill
            ref.append(dose)

            e = dose * 1.015                                       # +1.5% global bias
            # central hot spot: a 3x3 patch ~12% high (fails 3%/3mm)
            if abs(dx - 3.0) <= 1.0 and abs(dy + 2.0) <= 1.0:
                e = dose * 1.12
            ev.append(e)

    params = (spacing_mm, dd_percent, dta_mm, thresh_frac)
    return W, H, params, ref, ev


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic gamma-index sample.")
    ap.add_argument("--n", type=int, default=32, help="grid edge length (n x n)")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    W, H, params, ref, ev = build(args.n)

    # Format floats compactly but with enough digits that a round-trip through
    # float32 is stable (%.7g is exact for single precision).
    def fmt(vals):
        return " ".join(f"{v:.7g}" for v in vals)

    lines = [
        f"{W} {H}",
        " ".join(f"{p:g}" for p in params),
        fmt(ref),
        fmt(ev),
    ]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  ({W}x{H} dose pair; SYNTHETIC)")


if __name__ == "__main__":
    main()
