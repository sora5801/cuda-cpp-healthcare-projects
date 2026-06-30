#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the tiny committed SAXS sample
# ---------------------------------------------------------------------------
# Project 2.24 : SAXS / SANS Data-Driven Structure Modeling
#
# WHAT THIS MAKES (all SYNTHETIC -- no real experimental data; see data/README.md)
#   A small "protein-like" point cloud (a compact globular blob of atoms) plus a
#   synthetic experimental SAXS curve I_exp(q) computed FROM that same structure
#   with the exact Debye formula and a little Gaussian noise. Because the
#   experiment is generated from the model, the demo can show a near-perfect fit
#   (reduced chi^2 ~ 1) and a recovered Guinier Rg that matches the structure's
#   true geometric Rg -- an interpretable, self-checking sample (PATTERNS.md §6).
#
#   The C++ program (src/) recomputes the Debye curve on CPU and GPU and verifies
#   they agree; this script is just the data factory. The Debye math here is the
#   Python mirror of src/saxs_core.h, kept deliberately simple and commented.
#
# OUTPUT FORMAT (matches load_model() in src/reference_cpu.cpp):
#   line 1               : "n_atoms  n_q  true_rg"
#   next n_atoms lines   : "x y z f"            (Angstrom, Angstrom, Angstrom, electrons)
#   next n_q   lines     : "q  I_exp  sigma"    (1/Angstrom, intensity, error bar)
#   '#' starts a comment.
#
# USAGE
#   python make_synthetic.py [--out PATH] [--atoms N] [--seed S]
#   (defaults write the committed ../data/sample/saxs_sample.txt deterministically)
# ===========================================================================
import argparse
import math
import os
import random


def geometric_rg(coords, f):
    """True radius of gyration of the point set, weighted by scattering strength f.

    Rg^2 = sum_i f_i |r_i - r_cm|^2 / sum_i f_i, with the scattering-weighted
    center of mass r_cm. This is the structural number the demo's Guinier fit
    tries to recover from the curve.
    """
    fw = sum(f)
    cx = sum(fi * c[0] for fi, c in zip(f, coords)) / fw
    cy = sum(fi * c[1] for fi, c in zip(f, coords)) / fw
    cz = sum(fi * c[2] for fi, c in zip(f, coords)) / fw
    num = 0.0
    for fi, (x, y, z) in zip(f, coords):
        num += fi * ((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)
    return math.sqrt(num / fw)


def debye_intensity(q, coords, f):
    """I(q) = sum_i sum_j f_i f_j sinc(q r_ij) -- the Python mirror of saxs_core.h.

    Uses the same symmetric (diagonal + 2*upper-triangle) decomposition so the
    generated I_exp is consistent with what the C++ code forward-models.
    """
    n = len(coords)
    acc = sum(fi * fi for fi in f)                  # self terms: sinc(0)=1
    for i in range(n):
        xi, yi, zi = coords[i]
        fi = f[i]
        for j in range(i + 1, n):
            xj, yj, zj = coords[j]
            rij = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2)
            x = q * rij
            sinc = 1.0 - x * x / 6.0 if abs(x) < 1e-8 else math.sin(x) / x
            acc += 2.0 * fi * f[j] * sinc
    return acc


def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic SAXS sample.")
    here = os.path.dirname(os.path.abspath(__file__))
    default_out = os.path.join(here, "..", "data", "sample", "saxs_sample.txt")
    ap.add_argument("--out", default=default_out, help="output path")
    ap.add_argument("--atoms", type=int, default=40, help="number of point atoms")
    ap.add_argument("--nq", type=int, default=24, help="number of q points")
    ap.add_argument("--qmax", type=float, default=0.30, help="max q (1/Angstrom)")
    ap.add_argument("--noise", type=float, default=0.01, help="relative noise level")
    ap.add_argument("--seed", type=int, default=20260628, help="RNG seed (determinism)")
    args = ap.parse_args()

    # Fixed seed -> the committed sample is reproducible byte-for-byte.
    rng = random.Random(args.seed)

    # ---- build a compact globular blob: atoms inside a sphere of radius R ----
    # Rejection-sample points uniformly in a ball so the cloud is roughly
    # spherical (a crude stand-in for a folded globular protein domain).
    R = 18.0  # Angstrom; sphere radius (sets the molecule's overall size)
    coords = []
    while len(coords) < args.atoms:
        x = rng.uniform(-R, R)
        y = rng.uniform(-R, R)
        z = rng.uniform(-R, R)
        if x * x + y * y + z * z <= R * R:
            coords.append((x, y, z))

    # Per-atom scattering strength: use a constant ~6 electrons (carbon-like) so
    # the point-atom approximation is transparent. (Real codes use element- and
    # q-dependent form factors; THEORY.md explains the difference.)
    f = [6.0] * len(coords)

    true_rg = geometric_rg(coords, f)

    # ---- q grid (avoid q=0 exactly; start just above 0 for the Guinier fit) ----
    qmin = args.qmax / args.nq
    qs = [qmin + (args.qmax - qmin) * k / (args.nq - 1) for k in range(args.nq)]

    # ---- synthetic experiment: Debye curve + small relative Gaussian noise ----
    rows = []
    for q in qs:
        I = debye_intensity(q, coords, f)
        sigma = args.noise * I if I > 0 else 1.0           # 1% error bars
        I_noisy = I + rng.gauss(0.0, sigma)                # add measurement noise
        rows.append((q, I_noisy, sigma))

    # ---- write the sample ----
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        fh.write("# Synthetic SAXS sample for project 2.24 (NOT real experimental data).\n")
        fh.write("# Generated by scripts/make_synthetic.py; see data/README.md.\n")
        fh.write("# header: n_atoms n_q true_rg(Angstrom)\n")
        fh.write(f"{len(coords)} {len(qs)} {true_rg:.6f}\n")
        fh.write("# atoms: x y z f   (Angstrom, electrons)\n")
        for (x, y, z), fi in zip(coords, f):
            fh.write(f"{x:.6f} {y:.6f} {z:.6f} {fi:.6f}\n")
        fh.write("# curve: q I_exp sigma   (1/Angstrom, intensity, error)\n")
        for q, I, s in rows:
            fh.write(f"{q:.6f} {I:.6f} {s:.6f}\n")

    print(f"wrote {out}: {len(coords)} atoms, {len(qs)} q-points, true Rg = {true_rg:.3f} A")


if __name__ == "__main__":
    main()
