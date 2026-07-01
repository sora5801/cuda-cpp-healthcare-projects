#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the tiny synthetic MRF sample
# ---------------------------------------------------------------------------
# Project 4.21 : MR Fingerprinting Reconstruction
#
# WHAT THIS MAKES (and why it is SYNTHETIC)
#   A self-contained MR Fingerprinting problem small enough to commit and run
#   offline in milliseconds, engineered so the demo's result is meaningful and
#   verifiable (PATTERNS.md §6):
#     * A pseudorandom acquisition schedule (flip angles, TRs, TEs) of T frames.
#       We use a STRONG schedule -- large flip-angle swings that re-excite the
#       inversion-recovery transient (T1 leverage) and a widely-varying echo time
#       TE (T2 leverage) -- so different tissues produce genuinely DIFFERENT
#       fingerprint SHAPES (not just amplitudes, which normalization removes).
#     * A (T1, T2) dictionary of D atoms spanning plausible brain relaxation
#       times. We start from a candidate T1xT2 grid and GREEDILY PRUNE atoms that
#       are within a cosine THRESHOLD of an already-kept atom, so no two kept
#       atoms are near-collinear. This guarantees a WELL-SEPARATED dictionary --
#       the honest, reduced-scope teaching version of a real 10^5-atom dictionary
#       (THEORY.md §"Where this sits in the real world"). Separation matters
#       twice: it makes matching accurate, and it removes near-ties so the GPU's
#       SGEMM (which sums in a different float order than the CPU) can never flip
#       a voxel's argmax -- letting us verify GPU==CPU on the INTEGER index.
#     * V voxels, each DRAWN FROM A KNOWN dictionary atom: we simulate that
#       atom's fingerprint with the SAME closed-form Bloch model the C++ code
#       uses (mrf_core.h), scale it by a random proton density, and add a little
#       Gaussian noise. The known source atom is written as ground truth so the
#       C++ demo can report reconstruction accuracy.
#
#   THIS DATA IS SYNTHETIC. It is NOT from any scanner or patient, carries no
#   clinical meaning, and must never be used for diagnosis (CLAUDE.md §8).
#   Real MRF data (fastMRI MRF, IEEE DataPort, qMRI.org) is pointed to in
#   data/README.md and scripts/download_data.*; those require registration.
#
# DETERMINISM
#   A fixed RNG seed makes the committed sample reproducible: re-running this
#   script byte-for-byte regenerates data/sample/mrf_sample.txt. The Bloch model
#   here is a faithful Python port of mrf_core.h's simulate_atom so the C++
#   dictionary matches the signals the voxels were built from.
#
# USAGE
#   python scripts/make_synthetic.py                 # writes the default sample
#   python scripts/make_synthetic.py --T 200 --out data/sample/mrf_sample.txt
# ===========================================================================
import argparse
import math
import os
import random


def bloch_step(mz_before, alpha_rad, tr_ms, te_ms, t1_ms, t2_ms):
    """One frame of the simplified Bloch recursion -- the exact port of
    mrf::bloch_step in src/mrf_core.h. Returns (signal, mz_after)."""
    transverse = mz_before * math.sin(alpha_rad)
    signal = transverse * math.exp(-te_ms / t2_ms)
    mz_untipped = mz_before * math.cos(alpha_rad)
    mz_after = 1.0 - (1.0 - mz_untipped) * math.exp(-tr_ms / t1_ms)
    return signal, mz_after


def simulate_atom(alpha, tr, te, t1_ms, t2_ms):
    """Full length-T fingerprint for a tissue (T1, T2). Port of
    mrf::simulate_atom (starts from an inversion, Mz = -1)."""
    mz = -1.0
    out = []
    for a, r, e in zip(alpha, tr, te):
        s, mz = bloch_step(mz, a, r, e, t1_ms, t2_ms)
        out.append(s)
    return out


def l2norm(v):
    """L2-normalize a list to unit norm (port of mrf::normalize_inplace)."""
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic MRF sample.")
    ap.add_argument("--T", type=int, default=120, help="number of time frames")
    ap.add_argument("--V", type=int, default=64, help="number of voxels")
    ap.add_argument("--noise", type=float, default=0.005,
                    help="Gaussian noise std as a fraction of the signal norm")
    ap.add_argument("--sep_thresh", type=float, default=0.995,
                    help="greedy-prune cosine threshold (keep atoms below this)")
    ap.add_argument("--seed", type=int, default=421, help="RNG seed (project id)")
    ap.add_argument("--out", default=None, help="output path")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    T = args.T

    # --- acquisition schedule (STRONG: excites both T1 and T2 contrast) ------
    # Flip angles: large swings (up to ~80 deg) with a periodic re-excitation
    # envelope so the inversion-recovery transient (which depends on T1) keeps
    # being refreshed across the train instead of settling into a T1-blind
    # steady state. TRs jitter slightly. Echo times TE sweep from ~15 to ~120 ms
    # so that exp(-TE/T2) genuinely separates tissues across the T2 range -- this
    # is where the T2 discrimination comes from (see THEORY.md §"The math").
    alpha = []
    for t in range(T):
        env = math.exp(-((t % 40)) / 60.0)                 # periodic transient refresh
        deg = 5.0 + 75.0 * abs(math.sin(0.25 * t)) * env   # 5..80 deg
        alpha.append(math.radians(deg))
    tr = [16.0 + rng.uniform(-2.0, 2.0) for _ in range(T)]              # ms
    te = [15.0 + 105.0 * (0.5 - 0.5 * math.cos(0.21 * t)) for t in range(T)]  # 15..120 ms

    # --- dictionary: candidate T1 x T2 grid, then GREEDY SEPARATION PRUNE ----
    # Candidate relaxation-time grid (ms), physically ordered T2 << T1. We then
    # keep an atom only if it is < sep_thresh cosine to every atom already kept,
    # so the final dictionary is well separated (no near-collinear atoms). This
    # is the honest reduced-scope stand-in for a dense 10^5-atom dictionary.
    t1_cand = [250.0, 350.0, 500.0, 700.0, 1000.0, 1400.0, 2000.0, 2800.0, 4000.0]
    t2_cand = [25.0, 40.0, 65.0, 100.0, 160.0, 250.0, 400.0]
    candidates = [(a, b) for a in t1_cand for b in t2_cand if b <= 0.5 * a]

    dict_pairs = []          # kept (T1, T2) atoms
    kept_norm = []           # their unit-normalized fingerprints (for the prune test)
    for (a, b) in candidates:
        fp = l2norm(simulate_atom(alpha, tr, te, a, b))
        too_close = any(sum(x * y for x, y in zip(fp, kv)) >= args.sep_thresh
                        for kv in kept_norm)
        if not too_close:
            dict_pairs.append((a, b))
            kept_norm.append(fp)
    D = len(dict_pairs)

    # Un-normalized atoms (voxels are built by scaling these by a PD + noise).
    atoms = [simulate_atom(alpha, tr, te, a, b) for (a, b) in dict_pairs]

    # --- voxels: each drawn from a known atom + PD scale + noise -------------
    lines_sig = []
    for v in range(args.V):
        src = rng.randrange(D)               # the KNOWN ground-truth atom
        pd = rng.uniform(0.5, 2.0)           # random proton density / gain
        base = atoms[src]
        nrm = math.sqrt(sum(x * x for x in base)) or 1.0
        noise_std = args.noise * nrm
        sig = [pd * x + rng.gauss(0.0, noise_std) for x in base]
        toks = " ".join(f"{x:.6e}" for x in sig)
        lines_sig.append(f"{src} {toks}")

    # --- write the self-describing file --------------------------------------
    out = args.out or os.path.join(os.path.dirname(__file__), "..",
                                   "data", "sample", "mrf_sample.txt")
    out = os.path.normpath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("# Synthetic MR Fingerprinting sample (project 4.21). NOT clinical data.\n")
        f.write("# Generated by scripts/make_synthetic.py; see data/README.md.\n")
        f.write(f"T {T}\n")
        f.write(f"D {D}\n")
        f.write(f"V {args.V}\n")
        f.write("ALPHA " + " ".join(f"{a:.6e}" for a in alpha) + "\n")
        f.write("TR " + " ".join(f"{r:.6e}" for r in tr) + "\n")
        f.write("TE " + " ".join(f"{e:.6e}" for e in te) + "\n")
        f.write("DICT\n")
        for (a, b) in dict_pairs:
            f.write(f"{a:.6e} {b:.6e}\n")
        f.write("SIGNAL\n")
        for line in lines_sig:
            f.write(line + "\n")

    print(f"[make_synthetic] wrote {out}")
    print(f"[make_synthetic] T={T} frames, D={D} atoms (from {len(candidates)} "
          f"candidates, sep_thresh={args.sep_thresh}), V={args.V} voxels, "
          f"noise={args.noise} (SYNTHETIC)")


if __name__ == "__main__":
    main()
