#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate a synthetic DEER ensemble + target.
# ---------------------------------------------------------------------------
# Project 2.35 : Electron Paramagnetic Resonance (EPR/DEER) Constrained Modeling
#
# WHAT THIS BUILDS  (all SYNTHETIC -- no real protein, clearly labeled)
#   An ensemble of M MD "frames". Each frame carries two spin-label rotamer
#   clouds (siteA, siteB) of ROTAMERS endpoints each, in nanometres. A frame's
#   true mean spin-spin distance is set by placing the two clouds that far apart
#   along x; the rotamers themselves are a small Gaussian jitter cloud (the label
#   flexibility). So each frame's back-calculated P_m(r) is a narrow bump centred
#   on its design distance.
#
#   We engineer a KNOWN ANSWER (PATTERNS.md section 6): a chosen subset of frames
#   are "true" -- their design distance matches the experimental target's peak;
#   the rest are "decoy" frames at other distances. The committed target P_exp(r)
#   is a Gaussian centred on TRUE_R_NM. A correct max-entropy reweighting must
#   move population ONTO the true frames (and the demo checks exactly that).
#
# OUTPUT FORMAT (parsed by reference_cpu.cpp; documented in data/README.md):
#   line 1 : M ROTAMERS NBINS
#   per frame: a truth flag line (1/0), then ROTAMERS 'x y z' site-A lines,
#              then ROTAMERS 'x y z' site-B lines.
#   then NBINS target P_exp values (one per line; re-normalized on load).
#
# DETERMINISTIC: a fixed RNG seed -> the committed sample is reproducible, so the
# program's stdout is byte-identical every run.
#
# USAGE
#   python scripts/make_synthetic.py                 # write the committed sample
#   python scripts/make_synthetic.py --frames 200    # a bigger ensemble
# ===========================================================================
import argparse
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "deer_sample.txt"

# These MUST match src/deer_params.h (the loader validates them).
ROTAMERS = 24
NBINS = 50
R_MIN_NM = 1.5
R_BIN_NM = 0.1

# The synthetic "experimental" answer: true conformers sit at TRUE_R_NM, decoys
# are spread across other distances. The label jitter (rotamer cloud spread)
# broadens each frame's P_m(r) realistically.
TRUE_R_NM = 3.5            # design distance of the true conformational state (nm)
DECOY_RS = [2.2, 2.8, 4.4, 5.1, 5.7]   # decoy states elsewhere in the window (nm)
LABEL_JITTER_NM = 0.25     # Gaussian std of each rotamer endpoint about its site
TARGET_SIGMA_NM = 0.30     # width of the experimental target Gaussian (nm)


def make_cloud(rng, center):
    """A rotamer cloud: ROTAMERS endpoints jittered around a 3-D site center."""
    cx, cy, cz = center
    pts = []
    for _ in range(ROTAMERS):
        x = cx + rng.gauss(0.0, LABEL_JITTER_NM)
        y = cy + rng.gauss(0.0, LABEL_JITTER_NM)
        z = cz + rng.gauss(0.0, LABEL_JITTER_NM)
        pts.append((x, y, z))
    return pts


def frame_at_distance(rng, d_nm):
    """Two clouds whose centers are d_nm apart along x (mean spin-spin ~ d_nm)."""
    siteA = make_cloud(rng, (0.0, 0.0, 0.0))
    siteB = make_cloud(rng, (d_nm, 0.0, 0.0))
    return siteA, siteB


def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic DEER ensemble + target.")
    ap.add_argument("--frames", type=int, default=64, help="number of ensemble members M")
    ap.add_argument("--true-frac", type=float, default=0.25, help="fraction of frames at TRUE_R_NM")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    M = args.frames
    n_true = max(1, int(round(M * args.true_frac)))

    # Assign each frame a design distance: the first n_true are true matches at
    # TRUE_R_NM; the rest cycle through the decoy distances. (Order is fixed, so
    # the file -- and thus the program output -- is deterministic.)
    frames = []   # list of (truth_flag, siteA, siteB)
    for m in range(M):
        if m < n_true:
            d = TRUE_R_NM
            truth = 1
        else:
            d = DECOY_RS[(m - n_true) % len(DECOY_RS)]
            truth = 0
        siteA, siteB = frame_at_distance(rng, d)
        frames.append((truth, siteA, siteB))

    # Experimental target P_exp(r): a Gaussian centred on TRUE_R_NM over the same
    # bin grid the code uses. (Re-normalized to sum 1 when loaded.)
    target = []
    for b in range(NBINS):
        r = R_MIN_NM + (b + 0.5) * R_BIN_NM
        val = math.exp(-0.5 * ((r - TRUE_R_NM) / TARGET_SIGMA_NM) ** 2)
        target.append(val)

    # Emit the file.
    lines = [f"{M} {ROTAMERS} {NBINS}"]
    lines.append(f"# SYNTHETIC DEER ensemble (project 2.35). {n_true}/{M} frames are true matches at "
                 f"r={TRUE_R_NM} nm; rest are decoys. NOT real EPR data.")
    for (truth, siteA, siteB) in frames:
        lines.append(str(truth))
        for (x, y, z) in siteA:
            lines.append(f"{x:.5f} {y:.5f} {z:.5f}")
        for (x, y, z) in siteB:
            lines.append(f"{x:.5f} {y:.5f} {z:.5f}")
    lines.append("# experimental target P_exp(r), one value per bin (re-normalized on load)")
    for v in target:
        lines.append(f"{v:.8f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  "
          f"(M={M} frames, {n_true} true at r={TRUE_R_NM} nm, {ROTAMERS} rotamers/site; SYNTHETIC)")


if __name__ == "__main__":
    main()
