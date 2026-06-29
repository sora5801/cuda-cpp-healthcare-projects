#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate a synthetic noisy ECG waveform
# ---------------------------------------------------------------------------
# Project 7.10 : Physiological Signal & Waveform Analysis
#
# Builds a clearly-SYNTHETIC ECG-like signal: each heartbeat is a sum of
# Gaussians approximating the P, Q, R, S, T waves, repeated periodically, plus
# additive high-frequency noise and slow baseline wander. The demo low-pass
# filters this to remove the noise -- the classic 1-D convolution use case.
# Real waveforms come from PhysioNet/MIMIC (see download_data.*).
#
# OUTPUT (data/README.md format): "n" then n float samples, one per line.
#
# USAGE
#   python scripts/make_synthetic.py
#   python scripts/make_synthetic.py --n 8192 --noise 0.08
# ===========================================================================
import argparse
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "ecg_sample.txt"

# Wave templates relative to the R peak: (offset_samples, amplitude, width_samples)
WAVES = [
    (-60, 0.15, 12.0),   # P wave
    (-12, -0.10, 3.0),   # Q
    (0,   1.00, 2.5),    # R (tall, narrow)
    (12,  -0.25, 4.0),   # S
    (55,  0.30, 16.0),   # T
]


def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic noisy ECG waveform.")
    ap.add_argument("--n", type=int, default=2048, help="number of samples")
    ap.add_argument("--period", type=int, default=256, help="beat period (samples)")
    ap.add_argument("--noise", type=float, default=0.05, help="high-frequency noise std")
    ap.add_argument("--wander", type=float, default=0.10, help="baseline wander amplitude")
    ap.add_argument("--seed", type=int, default=3, help="RNG seed")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    n, T = args.n, args.period
    x = [0.0] * n
    # Place each beat's waves.
    for center in range(T // 2, n, T):
        for (off, amp, width) in WAVES:
            mu = center + off
            lo = max(0, int(mu - 4 * width))
            hi = min(n, int(mu + 4 * width) + 1)
            for i in range(lo, hi):
                x[i] += amp * math.exp(-0.5 * ((i - mu) / width) ** 2)
    # Add baseline wander (slow) + high-frequency noise.
    for i in range(n):
        x[i] += args.wander * math.sin(2.0 * math.pi * i / (3.0 * T))
        x[i] += rng.gauss(0.0, args.noise)

    lines = [str(n)] + [f"{v:.6f}" for v in x]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  (n={n}, {n // T} beats; SYNTHETIC ECG, seed={args.seed})")


if __name__ == "__main__":
    main()
