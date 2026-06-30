#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate a synthetic cryo-ET tilt series
# ---------------------------------------------------------------------------
# Project 2.31 : Cryo-EM Tilt-Series Alignment & Tomogram Reconstruction
#
# WHY THIS EXISTS
#   Real cryo-ET tilt series (EMPIAR) are large multi-GB stacks that need
#   registration and cannot be committed. So we generate a TINY, clearly-
#   SYNTHETIC stand-in with a KNOWN answer, so the demo runs offline and the
#   result is interpretable and verifiable (PATTERNS.md sec.6).
#
# WHAT IT MODELS  (a 2-D slice, the reduced-scope teaching case)
#   A "phantom" is a synthetic object of known shape. Its projection (the Radon
#   transform) of a uniform DISC is ANALYTIC and exact: a ray at detector offset
#   s through a disc of radius r centered at world (cx,cy) sees a chord of length
#   2*sqrt(r^2 - dx^2), where dx = s - (cx*cos(theta) + cy*sin(theta)). We sum
#   those chords over a few discs -> a deterministic projection, no rasterizing.
#
#   THREE cryo-ET-specific choices make this teach the project's ideas:
#     1. LIMITED, irregular tilt range: angles span only +-MAXTILT deg (default
#        +-60) -- the rest of Fourier space is the MISSING WEDGE. We also drop a
#        couple of angles to mimic an irregular acquisition schedule.
#     2. KNOWN per-projection DRIFT: each projection is shifted by a fixed,
#        reproducible integer number of detector bins (a sawtooth pattern). The
#        program's alignment step must RECOVER these shifts -- that is the
#        headline interpretable result on stdout.
#     3. A bright central disc so the reconstruction's max pixel lands at the
#        slice center, a quick visual sanity check.
#
# OUTPUT (data/README.md format):
#   header: "<n_tilts> <n_det> <ds> <img> <world_half>"
#   then n_tilts records: "<tilt_deg>  p_0 p_1 ... p_{n_det-1}".
#
# USAGE
#   python scripts/make_synthetic.py
#   python scripts/make_synthetic.py --maxtilt 60 --step 6 --det 129 --img 96
# ===========================================================================
import argparse
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent              # the project folder
OUT = ROOT / "data" / "sample" / "tilt_series_sample.txt"

# Phantom: list of discs (cx, cy, radius, density), world units (match world_half).
# A bright central disc (density 1.0) plus a few off-center inserts, including one
# "cold" (negative) feature, so the reconstruction has recognizable structure.
PHANTOM = [
    (0.00,  0.00, 0.16,  1.0),   # bright central disc -> reconstruction max
    (0.32,  0.10, 0.10,  0.6),
    (-0.28, 0.22, 0.09,  0.5),
    (0.05, -0.34, 0.08, -0.4),   # "cold" insert (negative density)
    (-0.18,-0.12, 0.06,  0.7),
]


def drift_for(k, n_tilts):
    """Deterministic, reproducible per-projection drift in DETECTOR BINS.

    A small sawtooth in [-3, +3] bins: the further from the central index, the
    larger the (signed) drift, like progressive stage creep across the series.
    The program's cross-correlation alignment must recover exactly -drift (it
    aligns each projection back onto the reference)."""
    mid = (n_tilts - 1) / 2.0
    # round(...) keeps it an integer bin shift; range stays within +-3.
    return int(round(3.0 * (k - mid) / mid)) if mid > 0 else 0


def main():
    ap = argparse.ArgumentParser(
        description="Generate a synthetic cryo-ET tilt series from a disc phantom.")
    ap.add_argument("--maxtilt", type=float, default=60.0,
                    help="tilt range is [-maxtilt, +maxtilt] degrees (missing wedge outside)")
    ap.add_argument("--step", type=float, default=6.0, help="nominal tilt increment (deg)")
    ap.add_argument("--det", type=int, default=129, help="detector bins per projection")
    ap.add_argument("--ds", type=float, default=0.012, help="detector bin spacing (world units)")
    ap.add_argument("--img", type=int, default=96, help="reconstruction slice side")
    ap.add_argument("--world-half", type=float, default=0.75, help="slice spans [-W,W]^2")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    # Build the (irregular) tilt schedule: -maxtilt..+maxtilt by `step`, then drop
    # two interior angles to mimic a real, non-uniform acquisition.
    n_steps = int(round(2 * args.maxtilt / args.step))
    angles = [(-args.maxtilt + i * args.step) for i in range(n_steps + 1)]
    # Drop two angles deterministically (indices chosen to leave a clean schedule).
    drop = {2, n_steps - 3} if n_steps >= 6 else set()
    angles = [a for i, a in enumerate(angles) if i not in drop]

    n_tilts = len(angles)
    n_det, ds = args.det, args.ds
    center = 0.5 * (n_det - 1)

    records = []
    for k, tilt_deg in enumerate(angles):
        theta = tilt_deg * math.pi / 180.0
        ct, st = math.cos(theta), math.sin(theta)
        drift = drift_for(k, n_tilts)              # known integer bin drift
        clean = [0.0] * n_det
        for j in range(n_det):
            s = (j - center) * ds
            val = 0.0
            for (cx, cy, r, d) in PHANTOM:
                c = cx * ct + cy * st              # disc center projected to detector
                dx = s - c
                if abs(dx) < r:
                    val += d * 2.0 * math.sqrt(r * r - dx * dx)  # chord * density
            clean[j] = val
        # Apply the drift: shift the clean projection RIGHT by `drift` bins, filling
        # vacated bins with 0. The loader sees only this drifted projection; the
        # program must estimate `drift` and undo it.
        drifted = [0.0] * n_det
        for j in range(n_det):
            sj = j - drift
            drifted[j] = clean[sj] if 0 <= sj < n_det else 0.0
        row = " ".join(f"{v:.6f}" for v in drifted)
        records.append(f"{tilt_deg:.1f} {row}")

    header = f"{n_tilts} {n_det} {ds} {args.img} {args.world_half}"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(header + "\n" + "\n".join(records) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  ({n_tilts} tilts in "
          f"[{angles[0]:.0f},{angles[-1]:.0f}] deg x {n_det} det, img={args.img}; "
          f"SYNTHETIC disc phantom with known drift)")


if __name__ == "__main__":
    main()
