#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate a synthetic SD-OCT B-scan
# ---------------------------------------------------------------------------
# Project 4.12 : Optical Coherence Tomography Processing (SD-OCT reconstruction)
#
# WHY THIS EXISTS
#   Real OCT raw spectra come with device-specific formats and licenses (OCTDL,
#   Duke DME, OCTA-500 -- see data/README.md and download_data.*). To keep the
#   demo runnable OFFLINE and INTERPRETABLE, we synthesise raw interferometric
#   spectra with a KNOWN layered structure, so the reconstruction has an obvious
#   right answer. The data is always LABELED synthetic (CLAUDE.md §8).
#
# THE FORWARD MODEL (how a real SD-OCT spectrometer forms a spectrum)
#   A single A-scan sees a set of reflectors at depths {z_r} with reflectivities
#   {R_r}. Each reflector contributes a cosine fringe in the spectrum whose
#   frequency is proportional to its depth:
#       I(k) = DC + sum_r  R_r * cos( 2*pi * z_r * k_norm  +  phi_disp(k) )
#   where k indexes the spectral (wavenumber) samples, k_norm = k/N maps a bin,
#   and phi_disp(k) = a2*(kc)^2 + a3*(kc)^3 is a DISPERSION phase error (kc is the
#   band-centred coordinate, matching src/oct_core.h::dispersion_phase). We INJECT
#   that dispersion here; the reconstruction REMOVES it (numerical dispersion
#   compensation), which is exactly why the reconstructed peaks are sharp. A
#   depth-z reflector shows up at FFT bin z after reconstruction -- the demo
#   recovers each A-scan's dominant reflector depth.
#
# OUTPUT FORMAT (data/README.md):
#   header:  "<n_ascan> <n_spec> <a2> <a3>"
#   body:    n_ascan rows, each n_spec raw spectrum floats.
#
# USAGE
#   python scripts/make_synthetic.py                 # data/sample/oct_bscan.txt
#   python scripts/make_synthetic.py --n-ascan 64 --n-spec 512
# ===========================================================================
import argparse
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "oct_bscan.txt"


def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic SD-OCT B-scan.")
    ap.add_argument("--n-ascan", type=int, default=32, help="A-scans (lateral pixels)")
    ap.add_argument("--n-spec", type=int, default=256, help="spectral samples per A-scan (FFT length, even)")
    ap.add_argument("--a2", type=float, default=18.0, help="injected 2nd-order dispersion coeff")
    ap.add_argument("--a3", type=float, default=9.0, help="injected 3rd-order dispersion coeff")
    ap.add_argument("--noise", type=float, default=0.02, help="per-sample noise std")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    A, N = args.n_ascan, args.n_spec
    if N % 2 != 0:
        raise SystemExit("--n-spec must be even")
    rng = random.Random(args.seed)

    # A gently curved "retina-like" surface plus two deeper layers, so the B-scan
    # has visible structure. Depth of the bright surface reflector for A-scan a:
    #   a smooth arc across the lateral field, kept well inside [0, N/2).
    def surface_depth(a):
        # arc from ~6 up to ~14 and back, deterministic in a.
        mid = 0.5 * (A - 1)
        arc = 8.0 - 6.0 * ((a - mid) / (mid + 1e-9)) ** 2   # peak in the middle
        return 6.0 + arc                                     # ~6..14

    def dispersion_phase(i):
        kc = (i - 0.5 * (N - 1)) / N
        return args.a2 * kc * kc + args.a3 * kc * kc * kc

    rows = []
    for a in range(A):
        z0 = surface_depth(a)
        # Three reflectors per A-scan: bright surface, two dimmer deeper layers.
        reflectors = [
            (z0,        1.0),          # bright surface (the strongest -> peak)
            (z0 + 18.0, 0.45),         # a deeper layer
            (z0 + 34.0, 0.30),         # a still-deeper layer
        ]
        row = []
        for i in range(N):
            k_norm = i / N
            val = 0.0
            for (z, R) in reflectors:
                # Fringe frequency proportional to depth z; +dispersion phase.
                val += R * math.cos(2.0 * math.pi * z * k_norm + dispersion_phase(i))
            val += 1.0                                   # DC / background offset
            val += rng.gauss(0.0, args.noise)            # detector noise
            row.append(val)
        rows.append(" ".join(f"{v:.6f}" for v in row))

    header = f"{A} {N} {args.a2:g} {args.a3:g}"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  ({A} A-scans x {N} spectral samples, "
          f"a2={args.a2:g} a3={args.a3:g}; SYNTHETIC SD-OCT, seed={args.seed})")


if __name__ == "__main__":
    main()
