#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate a synthetic pair of density maps
# ---------------------------------------------------------------------------
# Project 2.22 : Electron Density Map Analysis & Model Validation
#
# WHAT IT BUILDS  (a sample engineered to be INTERPRETABLE -- PATTERNS.md §6)
#   Two co-sampled n*n*n electron-density maps A and B:
#     * A "true" map of a few 3-D Gaussian "atoms" (blobs of density), the kind
#       of feature an electron-density map of a small molecule would show.
#     * Map A = true + low noise          (a clean reconstruction).
#     * Map B = true + MORE noise, and that extra noise is biased toward HIGH
#       spatial frequency (added as fine-grained, voxel-scale jitter).
#   Consequence -- which is the whole teaching point:
#     - RSCC (real-space correlation) is HIGH but < 1 (the maps mostly agree).
#     - FSC (Fourier shell correlation) is ~1 at LOW frequency (the big blobs
#       match) and DECAYS at HIGH frequency (the fine detail disagrees). The
#       shell where FSC crosses 0.143 is the "resolution" -- exactly what cryo-EM
#       validation reports. So the demo recovers a sensible resolution number.
#
#   Everything here is SYNTHETIC and labeled synthetic (CLAUDE.md §8). Real maps
#   come from EMDB / the PDB (see scripts/download_data.*).
#
# OUTPUT (data/README.md format):
#   header line: "<n> <voxel_angstrom>"
#   then n^3 floats for map A, then n^3 floats for map B (C-order: z,y,x).
#
# USAGE
#   python scripts/make_synthetic.py                 # default n=16, 2.0 A/voxel
#   python scripts/make_synthetic.py --n 32 --seed 7
# ===========================================================================
import argparse
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "map_sample.txt"

# A handful of Gaussian "atoms": (cx, cy, cz) centre in FRACTIONAL box coords,
# height, and sigma in voxels. These define the shared low-frequency structure.
ATOMS = [
    (0.30, 0.35, 0.40, 1.00, 1.8),
    (0.65, 0.55, 0.50, 0.85, 2.2),
    (0.50, 0.70, 0.60, 0.70, 1.5),
    (0.40, 0.45, 0.65, 0.60, 2.0),
]


def true_density(n):
    """The shared, noise-free density: sum of 3-D Gaussian blobs on an n^3 grid."""
    vol = [0.0] * (n * n * n)
    for (fx, fy, fz, h, sig) in ATOMS:
        cx, cy, cz = fx * n, fy * n, fz * n
        inv2s2 = 1.0 / (2.0 * sig * sig)
        for z in range(n):
            for y in range(n):
                for x in range(n):
                    r2 = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
                    vol[(z * n + y) * n + x] += h * math.exp(-r2 * inv2s2)
    return vol


def high_freq_jitter(n, rng, amp):
    """Voxel-scale checkerboard-modulated noise: noise * (-1)^(x+y+z). The sign
    flip puts most of this noise's power at the highest spatial frequency, so it
    degrades the HIGH-frequency FSC shells (the resolution-limiting term)."""
    out = [0.0] * (n * n * n)
    for z in range(n):
        for y in range(n):
            for x in range(n):
                s = 1.0 if ((x + y + z) % 2 == 0) else -1.0
                out[(z * n + y) * n + x] = amp * s * rng.gauss(0.0, 1.0)
    return out


def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic density-map pair.")
    ap.add_argument("--n", type=int, default=16, help="grid edge (cube is n^3)")
    ap.add_argument("--voxel", type=float, default=2.0, help="voxel size in Angstrom")
    ap.add_argument("--noise-a", type=float, default=0.02, help="map A low noise std")
    ap.add_argument("--noise-b", type=float, default=0.05, help="map B low noise std")
    ap.add_argument("--hf-b", type=float, default=0.12, help="map B high-freq jitter amp")
    ap.add_argument("--seed", type=int, default=22)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    n = args.n
    base = true_density(n)

    # Map A: true + small isotropic noise (a clean map).
    a = [v + rng.gauss(0.0, args.noise_a) for v in base]
    # Map B: true + larger isotropic noise + high-frequency jitter (loses detail).
    hf = high_freq_jitter(n, rng, args.hf_b)
    b = [base[i] + rng.gauss(0.0, args.noise_b) + hf[i] for i in range(n * n * n)]

    # Serialize: header, then all of A, then all of B (6 sig figs is plenty).
    lines = [f"{n} {args.voxel:g}"]
    lines.append(" ".join(f"{v:.6f}" for v in a))
    lines.append(" ".join(f"{v:.6f}" for v in b))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  "
          f"({n}^3 voxels x2 maps, {args.voxel:g} A/voxel; SYNTHETIC, seed={args.seed})")


if __name__ == "__main__":
    main()
