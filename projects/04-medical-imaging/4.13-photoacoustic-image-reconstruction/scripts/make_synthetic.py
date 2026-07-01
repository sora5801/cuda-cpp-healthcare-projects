#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate a synthetic photoacoustic dataset
# ---------------------------------------------------------------------------
# Project 4.13 : Photoacoustic Image Reconstruction
#
# THE FORWARD MODEL (what a PA scanner would measure)
#   A pulsed laser deposits energy in a few small optical absorbers inside
#   tissue. Each absorber instantly heats, expands, and launches an outgoing
#   ultrasound pulse. A ring of point sensors around the object records the
#   pressure vs. time. For a point source at position q with amplitude A, the
#   pressure that sensor s (at p_s) sees is a short bipolar pulse arriving at the
#   travel time
#        tau = |p_s - q| / c
#   We SYNTHESIZE each sensor's trace as the sum, over absorbers, of a short
#   POSITIVE Gaussian pulse centered at tau. A 1/dist geometric decay is applied
#   so nearer sensors see stronger signals -- a mild, realistic touch. (A real PA
#   pulse is BIPOLAR -- the time-derivative of the Gaussian -- which makes the raw
#   DAS image bipolar too; production reconstructs an envelope or applies the
#   universal-back-projection derivative. We use a unipolar pulse so the demo's
#   "brightest pixel = strongest source" check is clean and unambiguous; THEORY.md
#   §Where-this-sits explains the bipolar reality. This is a teaching stand-in for
#   a full k-Wave acoustic simulation; see data/README.md + download_data.*.)
#
#   Because delay-and-sum reconstruction sums the traces sampled at exactly these
#   travel times, the reconstruction will show BRIGHT PEAKS at the planted
#   absorber locations -- a known, verifiable answer (PATTERNS.md §6).
#
# OUTPUT FORMAT (matches src/reference_cpu.cpp::load_pa and data/README.md):
#   header : "<n_sensors> <n_samples> <dt> <c> <img> <world_half>"
#   then   : n_sensors lines "<sx> <sy>"           (sensor positions, metres)
#   then   : n_sensors lines of n_samples floats   (pressure traces)
#
# USAGE
#   python scripts/make_synthetic.py
#   python scripts/make_synthetic.py --sensors 64 --samples 400 --img 96
#
# The output is fully DETERMINISTIC (no RNG): identical every run, so the demo's
# expected_output.txt stays stable. Everything here is SYNTHETIC.
# ===========================================================================
import argparse
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "pa_sample.txt"

# Planted point absorbers: (x, y, amplitude) in metres / arbitrary units. These
# are the "ground truth" the reconstruction must recover. The first is the
# strongest, so the reported peak pixel should land on it. Kept well inside the
# sensor ring so every sensor sees every source. SYNTHETIC.
ABSORBERS = [
    (0.000,  0.000, 1.00),   # strong central source (the expected global peak)
    (0.005, -0.004, 0.70),   # off-center source
    (-0.004, 0.003, 0.55),   # another off-center source
]


def gaussian(t_over_dt, width_samples):
    """Unipolar Gaussian pulse sampled at integer offset t_over_dt from its
    center, with the given width (in samples). Positive-only so delay-and-sum
    reconstructs a clean bright peak at each source (no negative side-lobes).
    A physically-exact PA pulse is this Gaussian's time-derivative (bipolar);
    see the module header and THEORY.md."""
    a = t_over_dt / width_samples
    return math.exp(-a * a)


def main():
    ap = argparse.ArgumentParser(
        description="Generate a synthetic photoacoustic acquisition (ring array).")
    ap.add_argument("--sensors", type=int, default=64, help="number of ring sensors")
    ap.add_argument("--samples", type=int, default=512, help="time samples per sensor")
    ap.add_argument("--dt", type=float, default=5.0e-8, help="sample period [s] (20 MHz)")
    ap.add_argument("--c", type=float, default=1500.0, help="speed of sound [m/s]")
    ap.add_argument("--img", type=int, default=96, help="reconstruction image side")
    ap.add_argument("--world-half", type=float, default=0.010,
                    help="image spans [-W,W]^2 [m] (default +/-10 mm)")
    ap.add_argument("--radius", type=float, default=0.020,
                    help="sensor ring radius [m] (default 20 mm)")
    ap.add_argument("--pulse-width", type=float, default=2.0,
                    help="Ricker wavelet half-width in samples")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    ns, nt, dt, c = args.sensors, args.samples, args.dt, args.c
    R = args.radius

    # Sensor positions: evenly spaced on a circle of radius R around the origin.
    sensors = []
    for s in range(ns):
        ang = 2.0 * math.pi * s / ns
        sensors.append((R * math.cos(ang), R * math.sin(ang)))

    # Build each sensor's pressure trace by superposing the pulse from every
    # absorber, arriving at its own travel time and scaled by 1/dist.
    traces = []
    for (sx, sy) in sensors:
        trace = [0.0] * nt
        for (qx, qy, amp) in ABSORBERS:
            dist = math.hypot(sx - qx, sy - qy)      # |p_s - q| [m]
            tau = dist / c                           # travel time [s]
            center = tau / dt                        # arrival, in samples
            scale = amp / max(dist, 1e-6)            # geometric 1/r decay
            # Add the wavelet only over a local window (it decays fast) for speed.
            w = int(args.pulse_width)
            lo = max(0, int(center) - 6 * w - 2)
            hi = min(nt, int(center) + 6 * w + 3)
            for i in range(lo, hi):
                trace[i] += scale * gaussian(i - center, args.pulse_width)
        traces.append(trace)

    # Assemble the text file.
    header = f"{ns} {nt} {dt:.6e} {c:.1f} {args.img} {args.world_half:.6f}"
    lines = [header]
    for (sx, sy) in sensors:
        lines.append(f"{sx:.6e} {sy:.6e}")
    for trace in traces:
        lines.append(" ".join(f"{v:.6e}" for v in trace))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  ({ns} sensors x {nt} samples, "
          f"img={args.img}; SYNTHETIC ring-array PA data, {len(ABSORBERS)} absorbers)")


if __name__ == "__main__":
    main()
