#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate a synthetic multi-channel EEG window
# ---------------------------------------------------------------------------
# Project 8.03 : EEG/MEG Spectral Processing (cuFFT)
#
# Builds an 8-channel EEG window where each channel has a KNOWN dominant rhythm
# (so the band-power result is interpretable and the dominant band is obvious),
# plus low-level noise. With fs == n, frequency f Hz lands exactly on FFT bin f.
# Real EEG comes from PhysioNet/MNE (see download_data.*).
#
# OUTPUT (data/README.md format):
#   header: "n_ch n fs"  then n_ch rows of n float samples.
#
# USAGE
#   python scripts/make_synthetic.py
#   python scripts/make_synthetic.py --n 512 --fs 512
# ===========================================================================
import argparse
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "eeg_sample.txt"

# Per channel: list of (frequency_Hz, amplitude). The strongest sets the band.
#   delta~2  theta~6  alpha~10  beta~20  gamma~40
CHANNELS = [
    [(10, 1.0)],                 # ch0 alpha
    [(20, 1.0)],                 # ch1 beta
    [(6, 1.0)],                  # ch2 theta
    [(2, 1.0)],                  # ch3 delta
    [(40, 1.0)],                 # ch4 gamma
    [(11, 0.9), (18, 0.5)],      # ch5 alpha-dominant, some beta
    [(6, 0.9), (35, 0.4)],       # ch6 theta-dominant, some gamma
    [(22, 0.8), (9, 0.5)],       # ch7 beta-dominant, some alpha
]


def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic multi-channel EEG window.")
    ap.add_argument("--n", type=int, default=256, help="samples per channel (FFT length)")
    ap.add_argument("--fs", type=float, default=256.0, help="sampling rate (Hz)")
    ap.add_argument("--noise", type=float, default=0.10, help="per-sample noise std")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    n, fs = args.n, args.fs
    rows = []
    for comps in CHANNELS:
        row = []
        for t in range(n):
            v = 0.0
            for (f, amp) in comps:
                v += amp * math.sin(2.0 * math.pi * f * t / fs)
            v += rng.gauss(0.0, args.noise)
            row.append(v)
        rows.append(" ".join(f"{v:.6f}" for v in row))

    header = f"{len(CHANNELS)} {n} {fs:g}"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  ({len(CHANNELS)} channels x {n} samples, "
          f"fs={fs:g}; SYNTHETIC EEG, seed={args.seed})")


if __name__ == "__main__":
    main()
