#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate synthetic ultrasound RF echo data
# ---------------------------------------------------------------------------
# Project 4.6 : Ultrasound Beamforming (Delay-and-Sum)
#
# WHAT THIS MAKES (and WHY it is exactly verifiable)
#   We simulate the raw RF echoes a linear-array probe would record from a small
#   number of POINT SCATTERERS in tissue. The forward model is the textbook one:
#
#     * A scatterer at world point (sx, sz) reflects the transmit pulse.
#     * The echo recorded by element e (at (xe, 0)) is the transmit pulse,
#       delayed by the round-trip travel time
#           tau_e = ( dist(centre -> scatterer) + dist(scatterer -> element) )/c
#       and attenuated a little with range. The transmit leg uses a virtual
#       source at the array centre, matching beamform.h's DAS model exactly.
#     * The pulse itself is a Gaussian-windowed cosine at centre frequency f0
#       ("a few cycles") -- the standard short ultrasound pulse.
#
#   Because we KNOW where each scatterer is, a correct delay-and-sum beamformer
#   must focus its energy back onto that exact (sx, sz). The committed sample
#   uses ONE scatterer so the demo's "brightest pixel" lands on a known spot --
#   a self-checking, fully deterministic result (no randomness at all).
#
#   This is SYNTHETIC data (a simulator), labeled as such everywhere. Real RF
#   data comes from a scanner or a full-wave simulator (Field II / k-Wave); see
#   download_data.* and data/README.md.
#
# OUTPUT FORMAT (data/README.md):
#   header: "<n_elements> <n_samples> <nx> <nz> <fs> <c> <pitch> "
#           "<x_min> <z_min> <dx> <dz> <t0>"
#   then n_elements rows of n_samples floats (RF data, element-major).
#
# USAGE
#   python scripts/make_synthetic.py
#   python scripts/make_synthetic.py --elements 128 --samples 2048 --nx 192 --nz 192
# ===========================================================================
import argparse
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "rf_sample.txt"

# Scatterers as (x, z, amplitude) in metres. The committed sample keeps ONE so
# the focal spot is unambiguous; --extra adds two more for a richer image.
ONE_SCATTERER = [(0.004, 0.020, 1.0)]                       # 4 mm lateral, 20 mm deep
EXTRA_SCATTERERS = [(-0.006, 0.012, 0.8), (0.000, 0.030, 0.9)]


def main():
    ap = argparse.ArgumentParser(
        description="Generate synthetic ultrasound RF data from point scatterers.")
    ap.add_argument("--elements", type=int, default=64, help="transducer elements")
    ap.add_argument("--samples", type=int, default=384, help="RF samples per element")
    ap.add_argument("--nx", type=int, default=96, help="image width (lateral pixels)")
    ap.add_argument("--nz", type=int, default=96, help="image height (depth pixels)")
    ap.add_argument("--fs", type=float, default=40.0e6, help="RF sampling freq [Hz]")
    ap.add_argument("--c", type=float, default=1540.0, help="speed of sound [m/s]")
    ap.add_argument("--f0", type=float, default=5.0e6, help="pulse centre freq [Hz]")
    ap.add_argument("--pitch", type=float, default=0.3e-3, help="element spacing [m]")
    ap.add_argument("--t0", type=float, default=24.0e-6,
                    help="time of the FIRST recorded RF sample [s] (acquisition "
                         "window start; lets us skip the long empty pre-echo "
                         "stretch so the committed sample stays small)")
    ap.add_argument("--extra", action="store_true",
                    help="add two more scatterers (richer image, not used by demo)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    ne, ns, c, fs, f0 = args.elements, args.samples, args.c, args.fs, args.f0
    scatterers = ONE_SCATTERER + (EXTRA_SCATTERERS if args.extra else [])

    # ---- Image grid: a window in (x, z) that comfortably contains the probe
    #      aperture laterally and spans shallow-to-mid depth. -------------------
    x_min, z_min = -0.010, 0.005          # -10 mm.. , start 5 mm below probe
    dx = 0.020 / (args.nx - 1)            # 20 mm lateral field of view
    dz = 0.035 / (args.nz - 1)            # 5 mm .. 40 mm depth
    # The acquisition window starts at t0 (not 0): the transmit pulse and the
    # long quiet stretch before any echo returns are simply not recorded. This
    # is realistic (scanners record a depth-gated window) AND keeps the committed
    # sample small -- the loader/beamformer account for t0 everywhere.
    t0 = args.t0

    # Gaussian pulse envelope width: ~2 cycles of f0. sigma in seconds.
    sigma = 1.0 / (2.0 * math.pi * (f0 / 2.5))

    def element_x(e):
        # Array centred on x = 0 (matches beamform.h element_x()).
        return (e - 0.5 * (ne - 1)) * args.pitch

    def pulse(t):
        # Gaussian-windowed cosine: a short, band-limited ultrasound pulse.
        return math.exp(-(t * t) / (2.0 * sigma * sigma)) * math.cos(2.0 * math.pi * f0 * t)

    # ---- Synthesize the RF matrix ----------------------------------------
    rf = [[0.0] * ns for _ in range(ne)]
    for e in range(ne):
        xe = element_x(e)
        for (sx, sz, amp) in scatterers:
            # Round-trip time: virtual transmit from array centre (0,0) -> the
            # scatterer -> this element. Identical geometry to beamform.h's DAS.
            d_tx = math.hypot(sx, sz)                 # centre -> scatterer
            d_rx = math.hypot(sx - xe, sz)            # scatterer -> element
            tau = (d_tx + d_rx) / c                   # seconds
            # Gentle range attenuation so deeper echoes are a touch weaker.
            atten = amp / (1.0 + sz / 0.02)
            center_sample = (tau - t0) * fs
            # Deposit the pulse only near its arrival (a few sigma wide window)
            # to keep the file sparse and the simulation cheap.
            half = int(6.0 * sigma * fs) + 1
            i0 = max(0, int(round(center_sample)) - half)
            i1 = min(ns, int(round(center_sample)) + half)
            for i in range(i0, i1):
                t = (i - center_sample) / fs          # time relative to arrival
                rf[e][i] += atten * pulse(t)

    # ---- Write header + RF rows ------------------------------------------
    header = (f"{ne} {ns} {args.nx} {args.nz} "
              f"{fs:.6g} {c:.6g} {args.pitch:.6g} "
              f"{x_min:.6g} {z_min:.6g} {dx:.8g} {dz:.8g} {t0:.6g}")
    rows = [" ".join(f"{v:.6f}" for v in rf[e]) for e in range(ne)]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  ({ne} elements x {ns} samples, "
          f"img {args.nx}x{args.nz}; {len(scatterers)} SYNTHETIC point scatterer(s))")
    print("[make_synthetic] scatterer(s) at (x,z) mm: "
          + ", ".join(f"({1000*sx:.1f},{1000*sz:.1f})" for sx, sz, _ in scatterers))


if __name__ == "__main__":
    main()
