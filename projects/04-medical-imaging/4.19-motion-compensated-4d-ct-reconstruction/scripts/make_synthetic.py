#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate a synthetic 4D-CT (breathing) sinogram
# ---------------------------------------------------------------------------
# Project 4.19 : Motion-Compensated 4D-CT Reconstruction (2-D teaching version)
#
# WHY THIS EXISTS
#   Real 4D-CT lung data (DIR-Lab, TCIA, POPI) is large and licensed; it cannot
#   be redistributed in this repo. So we generate a clearly-SYNTHETIC stand-in
#   that (a) matches the loader's layout and (b) is engineered so the demo is
#   INTERPRETABLE: a small structure BREATHES (moves per phase), so the naive
#   reconstruction visibly blurs and motion compensation visibly sharpens it.
#
# THE FORWARD MODEL (why the sinogram looks the way it does)
#   The phantom is a sum of uniform DISCS. The line integral (Radon transform) of
#   a disc is ANALYTIC and exact: a ray at detector offset s crosses a disc of
#   radius r centered at projected position c with chord length
#       2 * sqrt(r^2 - (s - c)^2)   (0 if |s - c| >= r),
#   so we never rasterize -- the data is deterministic and noise-free.
#
#   To make it "4D" the phantom MOVES with breathing. For phase p we displace
#   every disc center by the SAME Deformation Vector Field the reconstruction
#   uses (mc4dct.h::dvf_at), so the motion model is consistent end to end:
#       (cx, cy) -> (cx + dvf_x, cy + dvf_y)   during phase p.
#   Phase 0 has zero motion (m(0)=0) and is the REFERENCE the MCR recovers.
#
#   Angles are assigned so that (i) the union over all phases tiles a full half
#   turn [0, pi), and (ii) each individual phase samples a SPARSE, interleaved
#   subset -> severe per-phase under-sampling, exactly like real phase-binned
#   4D-CT. Global index k = p*n_ang_phase + a has angle theta_k = k*pi/total.
#
# OUTPUT (data/README.md format):
#   header: "<img> <n_det> <n_phases> <n_ang_phase> <ds> <world_half> <amp>"
#   then (n_phases*n_ang_phase) rows of n_det floats (the sinogram), phase-major.
#
# USAGE
#   python scripts/make_synthetic.py
#   python scripts/make_synthetic.py --phases 10 --ang-phase 12 --det 129
# ===========================================================================
import argparse
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "sinogram4d_sample.txt"

# Phantom in REFERENCE (phase-0) coordinates: list of (cx, cy, radius, density).
#   The star of the demo is a single bright "nodule" placed LOW in the field,
#   where the diaphragm-like DVF pushes it the most. Under naive reconstruction
#   its energy smears across every phase's position, so its reconstructed PEAK
#   drops well below its true density (1.0). Motion compensation re-aligns all
#   phases, concentrating the energy back into a point -> the peak recovers to
#   ~1.0. That "peak recovery toward the true density" is the headline result.
#
#   A second, small reference nodule sits near the rotation axis (little motion),
#   so the image is not empty and the learner can compare a moving vs a nearly
#   static feature. Densities/positions chosen so the MOVING nodule is the global
#   peak (the quantity main.cu reports). Coordinates in world_half units.
PHANTOM = [
    (0.00, -0.35, 0.11, 1.0),    # MOVING nodule (low in field -> large DVF)
    (0.30,  0.30, 0.06, 0.5),    # small near-corner reference structure (context)
]


def phase_motion(p, n_phases):
    """Raised-cosine breathing amount in [0,1]; matches mc4dct.h::phase_motion.
    m(0)=0 (reference), m(P/2)=1 (peak displacement)."""
    ang = 2.0 * math.pi * p / n_phases
    return 0.5 * (1.0 - math.cos(ang))


def dvf(wx, wy, amp, world_half, p, n_phases):
    """Reference->phase displacement; MUST mirror mc4dct.h::dvf_at exactly.
    Vertical diaphragm-like push (bigger toward the bottom) + small lateral
    expansion, scaled by the breathing amount m(p)."""
    m = phase_motion(p, n_phases)
    nx = wx / world_half if world_half > 0 else 0.0
    ny = wy / world_half if world_half > 0 else 0.0
    dy = amp * m * (0.5 - 0.5 * ny)
    dx = 0.25 * amp * m * nx
    return dx, dy


def main():
    ap = argparse.ArgumentParser(
        description="Generate a synthetic breathing (4D) CT sinogram from a disc phantom.")
    ap.add_argument("--img", type=int, default=96, help="reconstruction image side (pixels)")
    ap.add_argument("--det", type=int, default=129, help="detector bins per projection")
    ap.add_argument("--phases", type=int, default=8, help="number of breathing phases P")
    ap.add_argument("--ang-phase", type=int, default=10, help="projection angles PER phase")
    ap.add_argument("--ds", type=float, default=0.02, help="detector bin spacing (world units)")
    ap.add_argument("--world-half", type=float, default=1.0, help="image spans [-W,W]^2")
    ap.add_argument("--amp", type=float, default=0.50, help="breathing motion amplitude (world units)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    P, A = args.phases, args.ang_phase
    n_det, ds, W, amp = args.det, args.ds, args.world_half, args.amp
    total = P * A
    center = 0.5 * (n_det - 1)

    rows = []
    # Phase-major storage: for each phase p, for each of its A angles a.
    for p in range(P):
        for a in range(A):
            k = p * A + a                        # global projection index
            theta = math.pi * k / total          # uniform half-turn across phases
            ct, st = math.cos(theta), math.sin(theta)
            row = []
            for j in range(n_det):
                s = (j - center) * ds
                val = 0.0
                for (cx, cy, r, d) in PHANTOM:
                    # Displace this disc's center by the phase-p DVF, evaluated at
                    # the disc's reference position (a rigid shift of the whole
                    # disc -- fine for a small disc under a smooth field).
                    dx, dy = dvf(cx, cy, amp, W, p, P)
                    px, py = cx + dx, cy + dy
                    c = px * ct + py * st          # deformed center projected onto detector
                    off = s - c
                    if abs(off) < r:
                        val += d * 2.0 * math.sqrt(r * r - off * off)  # chord * density
                row.append(val)
            rows.append(" ".join(f"{v:.6f}" for v in row))

    header = f"{args.img} {n_det} {P} {A} {ds} {W} {amp}"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  "
          f"({P} phases x {A} angles = {total} proj, {n_det} det, img={args.img}; "
          f"SYNTHETIC breathing disc phantom)")


if __name__ == "__main__":
    main()
