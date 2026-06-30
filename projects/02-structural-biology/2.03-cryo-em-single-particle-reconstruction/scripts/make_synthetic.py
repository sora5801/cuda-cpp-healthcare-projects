#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic cryo-EM sample dataset
# ---------------------------------------------------------------------------
# Project 2.3 : Cryo-EM Single-Particle Reconstruction  (reduced-scope, 2D)
#
# WHY SYNTHETIC
#   Real cryo-EM particle stacks (EMPIAR) and density maps (EMDB) are large and
#   sometimes access-controlled (see scripts/download_data.* and data/README.md).
#   To keep the demo OFFLINE and reproducible we deterministically generate a
#   clearly-SYNTHETIC 2D analogue of the single-particle problem:
#
#     * a "molecule" = a small 2D phantom (a few Gaussian blobs in an asymmetric
#       arrangement, so its projections look DIFFERENT at different angles --
#       which is exactly what lets projection matching recover orientation);
#     * a REFERENCE BANK = the phantom's 1D parallel-beam projections at N_ANGLES
#       evenly spaced angles over [0, pi);
#     * N PARTICLES = each is a reference projection at a randomly chosen angle
#       with additive Gaussian noise (mimicking cryo-EM's brutal SNR), and we
#       record the true angle index so main.cu can REPORT recovery accuracy.
#
#   The forward projector here is a byte-for-byte port of project_sample() in
#   src/reference_cpu.h (same centre, same bilinear sampling, same ray walk), so
#   the templates the C++ matcher scores against are geometrically consistent
#   with how it would itself project. A fixed RNG seed makes the file stable, so
#   demo/expected_output.txt never drifts.
#
# OUTPUT FORMAT (data/README.md):
#   line 1            : "n IMG_SIZE N_ANGLES PROJ_LEN"   (header + geometry check)
#   IMG_SIZE*IMG_SIZE : the ground-truth density (row-major floats)
#   N_ANGLES*PROJ_LEN : the reference projection bank (row-major)
#   per particle      : one int (true angle index) then PROJ_LEN profile floats
#
# USAGE
#   python scripts/make_synthetic.py                 # default n=200 particles
#   python scripts/make_synthetic.py --n 100000      # "near scale" stress set
# ===========================================================================
import argparse
import math
import random
from pathlib import Path

# These MUST match the constexpr geometry in src/reference_cpu.h.
IMG_SIZE = 64
PROJ_LEN = IMG_SIZE
N_ANGLES = 60
PI = 3.14159265358979323846

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "cryoem_sample.txt"


def ref_angle(a):
    """theta_a = a*pi/N_ANGLES -- identical to ref_angle() in reference_cpu.h."""
    return a * PI / N_ANGLES


def make_phantom():
    """Build the synthetic 'molecule': a sum of Gaussian blobs placed
    asymmetrically so that projections at different angles are distinguishable.
    Returns a row-major list of IMG_SIZE*IMG_SIZE floats."""
    centre = (IMG_SIZE - 1) * 0.5
    # (cx, cy, sigma, amplitude) for each blob, in pixels relative to centre.
    # The asymmetry (blobs off-axis, different sizes) is deliberate: a radially
    # symmetric phantom would project identically at every angle and orientation
    # recovery would be impossible -- a real lesson kept in THEORY/exercises.
    blobs = [
        (-10.0, -6.0, 5.0, 1.0),
        (8.0,  -2.0, 3.5, 0.8),
        (2.0,  10.0, 4.0, 0.9),
        (-4.0,  4.0, 2.5, 0.6),
    ]
    img = [0.0] * (IMG_SIZE * IMG_SIZE)
    for py in range(IMG_SIZE):
        for px in range(IMG_SIZE):
            dx = px - centre
            dy = py - centre
            v = 0.0
            for (cx, cy, sig, amp) in blobs:
                r2 = (dx - cx) ** 2 + (dy - cy) ** 2
                v += amp * math.exp(-r2 / (2.0 * sig * sig))
            img[py * IMG_SIZE + px] = v
    return img


def sample_at(img, xi, yi):
    """Bounds-checked pixel read (zero outside) -- ports sample_at() in C++."""
    if xi < 0 or xi >= IMG_SIZE or yi < 0 or yi >= IMG_SIZE:
        return 0.0
    return img[yi * IMG_SIZE + xi]


def project_sample(img, theta, s):
    """1D projection value at detector sample s, view angle theta. A direct port
    of project_sample() in src/reference_cpu.h (same geometry + bilinear walk)."""
    centre = (IMG_SIZE - 1) * 0.5
    t = s - centre
    ct = math.cos(theta)
    st = math.sin(theta)
    acc = 0.0
    for k in range(PROJ_LEN):
        u = k - centre
        x = centre + (t * ct - u * st)
        y = centre + (t * st + u * ct)
        x0 = math.floor(x)
        y0 = math.floor(y)
        fx = x - x0
        fy = y - y0
        v00 = sample_at(img, x0,     y0)
        v10 = sample_at(img, x0 + 1, y0)
        v01 = sample_at(img, x0,     y0 + 1)
        v11 = sample_at(img, x0 + 1, y0 + 1)
        top = v00 * (1.0 - fx) + v10 * fx
        bot = v01 * (1.0 - fx) + v11 * fx
        acc += top * (1.0 - fy) + bot * fy
    return acc


def project_all(img, theta):
    return [project_sample(img, theta, s) for s in range(PROJ_LEN)]


def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic 2D cryo-EM dataset.")
    ap.add_argument("--n", type=int, default=200, help="number of particles")
    ap.add_argument("--noise", type=float, default=0.15,
                    help="Gaussian noise stddev as a fraction of the profile RMS")
    ap.add_argument("--seed", type=int, default=11, help="RNG seed (determinism)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # 1. The ground-truth molecule.
    img = make_phantom()

    # 2. The reference projection bank (one profile per angle).
    refs = [project_all(img, ref_angle(a)) for a in range(N_ANGLES)]

    # A representative profile RMS to scale the noise to a realistic SNR.
    flat = [v for prof in refs for v in prof]
    rms = math.sqrt(sum(v * v for v in flat) / len(flat))
    sigma = args.noise * rms

    # 3. The particles: a random reference angle + additive Gaussian noise.
    particles = []
    for _ in range(args.n):
        a = rng.randrange(N_ANGLES)                       # true (hidden) angle
        prof = [refs[a][s] + rng.gauss(0.0, sigma) for s in range(PROJ_LEN)]
        particles.append((a, prof))

    # 4. Serialize in the loader's format.
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    parts = [f"{args.n} {IMG_SIZE} {N_ANGLES} {PROJ_LEN}"]
    parts.append(" ".join(f"{v:.6f}" for v in img))
    for prof in refs:
        parts.append(" ".join(f"{v:.6f}" for v in prof))
    for (a, prof) in particles:
        parts.append(str(a) + " " + " ".join(f"{v:.6f}" for v in prof))
    out.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {out}  (n={args.n}, {IMG_SIZE}x{IMG_SIZE} image, "
          f"{N_ANGLES} angles; SYNTHETIC, seed={args.seed}, noise={args.noise})")


if __name__ == "__main__":
    main()
