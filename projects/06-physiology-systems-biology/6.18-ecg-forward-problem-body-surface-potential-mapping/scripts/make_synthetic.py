#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic ECG-forward sample
# ---------------------------------------------------------------------------
# Project 6.18 : ECG Forward Problem & Body-Surface Potential Mapping
#
# WHY THIS EXISTS
#   The real datasets this project points at (PhysioNet, the EDGAR body-surface
#   potential database, the Visible Human torso) either require registration or
#   ship full 3-D torso meshes far too large to commit. So we generate a tiny,
#   clearly-SYNTHETIC stand-in that (a) matches the loader's format and (b) has a
#   known answer the demo can recover: the electrode nearest the strongest,
#   most-swinging cardiac source must record the largest peak-to-peak deflection.
#
#   NOTHING here is patient data. It is a deterministic geometric toy: a
#   cylindrical "torso" with electrodes on its surface and a few current dipoles
#   inside standing in for the heart's activation sequence.
#
# THE MODEL WE WRITE (all lengths in metres; see data/README.md for the format)
#   * L electrodes evenly spaced on a ring around a cylindrical torso surface.
#   * S dipole sources at fixed positions inside (a small "heart" cluster),
#     each with a fixed unit-ish direction and a time-varying STRENGTH.
#   * T frames of a synthetic activation: each source fires a smooth Gaussian
#     "bump" at a staggered time (a crude depolarization sweep). One source is
#     deliberately made the strongest so the ground-truth peak lead is definite.
#
#   Deterministic: no RNG, only closed-form math, so the committed sample and its
#   expected_output.txt never drift.
#
# USAGE
#   python scripts/make_synthetic.py                    # default tiny sample
#   python scripts/make_synthetic.py --L 12 --S 4 --T 40 --out data/sample/ecg_sample.txt
# ===========================================================================
import argparse
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent            # the project folder
OUT = ROOT / "data" / "sample" / "ecg_sample.txt"


def build_model(L, S, T):
    """Return (electrodes, src_pos, src_dir, strengths) for the synthetic torso.
    electrodes : list of (x,y,z); src_pos/src_dir : lists of (x,y,z);
    strengths  : list of S rows, each a list of T floats (the X matrix)."""
    torso_radius = 0.15      # m: cylindrical torso ~30 cm across
    torso_z = 0.0            # m: electrodes on one transverse ring (chest plane)

    # --- electrodes evenly around the torso ring -------------------------
    electrodes = []
    for e in range(L):
        ang = 2.0 * math.pi * e / L
        electrodes.append((torso_radius * math.cos(ang),
                           torso_radius * math.sin(ang),
                           torso_z))

    # --- dipole sources: a small cluster offset toward the "left chest" --
    # Place sources on a tiny inner ring near (x>0) so a specific electrode is
    # unambiguously the closest to the strongest source (a clean ground truth).
    heart_radius = 0.03      # m: sources sit ~3 cm from the torso axis
    heart_cx = 0.04          # m: cluster centre shifted toward +x (electrode 0)
    src_pos, src_dir = [], []
    for s in range(S):
        ang = 2.0 * math.pi * s / S
        px = heart_cx + heart_radius * math.cos(ang)
        py = heart_radius * math.sin(ang)
        pz = 0.0
        src_pos.append((px, py, pz))
        # Direction: point roughly radially outward from the torso axis so each
        # dipole projects onto the nearby electrodes (a plausible depolarization
        # orientation). Normalized in the loader, but we keep it near unit here.
        norm = math.sqrt(px * px + py * py) or 1.0
        src_dir.append((px / norm, py / norm, 0.3))

    # --- activation strengths: staggered Gaussian bumps ------------------
    # Source s fires a smooth bump centred at frame peak_t[s]; source 0 is the
    # strongest (amplitude 1.0) so the electrode nearest it is the ground-truth
    # peak lead. Others are weaker and later (a crude activation sweep).
    strengths = []
    width = max(2.0, T / 8.0)     # bump width in frames
    for s in range(S):
        amp = 1.0 if s == 0 else 0.4       # source 0 dominates on purpose
        peak_t = (T * (s + 1)) / (S + 1)   # staggered firing times
        row = []
        for t in range(T):
            g = amp * math.exp(-0.5 * ((t - peak_t) / width) ** 2)
            row.append(g)
        strengths.append(row)
    return electrodes, src_pos, src_dir, strengths


def fmt(v):
    """Fixed 6-decimal formatting so the committed sample bytes are stable."""
    return f"{v:.6f}"


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic ECG-forward sample.")
    ap.add_argument("--L", type=int, default=8, help="number of electrodes")
    ap.add_argument("--S", type=int, default=3, help="number of dipole sources")
    ap.add_argument("--T", type=int, default=24, help="number of time frames")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    L, S, T = args.L, args.S, args.T
    electrodes, src_pos, src_dir, strengths = build_model(L, S, T)

    lines = []
    lines.append("# Synthetic ECG forward-problem sample (NOT patient data).")
    lines.append("# Format: header 'L S T', then L electrode xyz, S source xyz,")
    lines.append("#         S direction xyz, then S rows of T source strengths.")
    lines.append(f"{L} {S} {T}")
    lines.append("# electrode positions (metres) x y z")
    for (x, y, z) in electrodes:
        lines.append(f"{fmt(x)} {fmt(y)} {fmt(z)}")
    lines.append("# source (dipole) positions (metres) x y z")
    for (x, y, z) in src_pos:
        lines.append(f"{fmt(x)} {fmt(y)} {fmt(z)}")
    lines.append("# source (dipole) directions (unit-ish) dx dy dz")
    for (x, y, z) in src_dir:
        lines.append(f"{fmt(x)} {fmt(y)} {fmt(z)}")
    lines.append("# source strength time series: S rows x T frames")
    for row in strengths:
        lines.append(" ".join(fmt(v) for v in row))

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {outp}  (L={L}, S={S}, T={T}; SYNTHETIC)")


if __name__ == "__main__":
    main()
