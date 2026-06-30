#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Forward-model a cryo-EM micrograph with a
#                                 KNOWN defocus (so the demo can recover it)
# ---------------------------------------------------------------------------
# Project 2.11 : Cryo-EM CTF Estimation & Particle Picking
#
# WHY SYNTHETIC (CLAUDE.md §8)
#   Real cryo-EM micrographs (EMPIAR) are large, gigabytes per dataset, and need a
#   download. For an OFFLINE, deterministic, *verifiable* demo we generate a tiny
#   image whose true defocus we control, so the fitter's answer can be checked
#   against the ground truth. The data is clearly labeled synthetic everywhere.
#
# HOW THE FORWARD MODEL WORKS (this is the physics the fitter inverts)
#   A weak-phase object's recorded image, in Fourier space, is the object's
#   spectrum multiplied by the CTF:  I_hat(k) = O_hat(k) * CTF(k) + noise.
#   We take O_hat to be WHITE (flat random spectrum, i.e. a featureless specimen),
#   multiply by CTF(k) for a chosen defocus, inverse-FFT to get the real-space
#   image, and add Gaussian shot noise. The resulting image's POWER spectrum is
#   |CTF(k)|^2 * (white) -> it shows genuine Thon rings at the chosen defocus.
#   That is exactly the situation CTF estimation faces in the wild.
#
#   We deliberately use a SMALL N (default 96) so the O(N^4) CPU reference DFT in
#   reference_cpu.cpp runs in well under a second. Real micrographs are 4k x 4k and
#   are FFT'd, never naive-DFT'd -- that is the whole point of the GPU path.
#
# USAGE
#   python scripts/make_synthetic.py                 # writes data/sample/micrograph_sample.txt
#   python scripts/make_synthetic.py --n 128 --defocus 18000 --out my.txt
#
# OUTPUT FORMAT (matches load_micrograph in src/reference_cpu.h):
#   line 1:  n pixel_size lambda cs amp_contrast true_dz
#   body  :  n*n floats, row-major.
# ===========================================================================
import argparse
import math
import os

import numpy as np


def electron_wavelength_A(kv: float) -> float:
    """Relativistic electron wavelength (Angstrom) at accelerating voltage `kv`
    (kilovolts). The standard formula:  lambda = h / sqrt(2 m e V (1 + e V / 2 m c^2)).
    At 300 kV this returns ~0.0197 A."""
    h = 6.62607015e-34      # Planck (J s)
    m = 9.1093837015e-31    # electron mass (kg)
    e = 1.602176634e-19     # elementary charge (C)
    c = 299792458.0         # speed of light (m/s)
    V = kv * 1.0e3          # volts
    lam_m = h / math.sqrt(2.0 * m * e * V * (1.0 + e * V / (2.0 * m * c * c)))
    return lam_m * 1.0e10   # metres -> Angstrom


def ctf_2d(n: int, pixel_size: float, lam: float, cs: float, ac: float, dz: float):
    """Return the (signed) CTF sampled on the n x n FFT frequency grid.
    Mirrors src/ctf_model.h ctf_value() so the synthetic rings sit where the
    fitter looks for them."""
    # Frequency axis (cycles / Angstrom): np.fft.fftfreq gives cycles/pixel,
    # divide by pixel_size to get cycles/Angstrom.
    f = np.fft.fftfreq(n, d=pixel_size)            # (n,)
    fu, fv = np.meshgrid(f, f, indexing="xy")      # (n,n)
    k2 = fu * fu + fv * fv                          # k^2  (1/A^2)
    k4 = k2 * k2
    phi = math.asin(ac)                             # amplitude-contrast phase
    chi = math.pi * lam * dz * k2 \
        - 0.5 * math.pi * cs * lam**3 * k4 \
        + phi
    return np.sin(chi)


def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic cryo-EM micrograph with a known defocus.")
    ap.add_argument("--n", type=int, default=96, help="image side length (pixels, power-of-two-ish; default 96)")
    ap.add_argument("--pixel", type=float, default=1.0, help="pixel size on specimen (A/pixel)")
    ap.add_argument("--kv", type=float, default=300.0, help="accelerating voltage (kV)")
    ap.add_argument("--cs", type=float, default=2.7e7, help="spherical aberration (A); 2.7e7 A = 2.7 mm")
    ap.add_argument("--ac", type=float, default=0.10, help="amplitude contrast fraction [0,1]")
    ap.add_argument("--defocus", type=float, default=15000.0, help="TRUE defocus (A); default 1.5 um")
    ap.add_argument("--noise", type=float, default=0.5, help="Gaussian noise std (relative to signal)")
    ap.add_argument("--seed", type=int, default=2112, help="RNG seed (deterministic sample)")
    ap.add_argument("--out", default=None, help="output path (default data/sample/micrograph_sample.txt)")
    args = ap.parse_args()

    lam = electron_wavelength_A(args.kv)
    rng = np.random.default_rng(args.seed)

    # White object spectrum: a flat-magnitude, random-phase complex field. Real
    # image => enforce Hermitian symmetry implicitly by building in real space and
    # FFTing. Simplest robust route: start from white real-space noise (flat
    # expected spectrum), FFT, multiply by CTF, inverse-FFT, take the real part.
    obj = rng.standard_normal((args.n, args.n))
    obj_hat = np.fft.fft2(obj)
    ctf = ctf_2d(args.n, args.pixel, lam, args.cs, args.ac, args.defocus)
    img_hat = obj_hat * ctf                          # apply the transfer function
    img = np.fft.ifft2(img_hat).real                 # back to real space

    # Normalize to unit std, then add detector/shot noise (also unit-scaled).
    img = (img - img.mean()) / (img.std() + 1e-12)
    img = img + args.noise * rng.standard_normal((args.n, args.n))

    out = args.out
    if out is None:
        here = os.path.dirname(os.path.abspath(__file__))
        out = os.path.join(here, "..", "data", "sample", "micrograph_sample.txt")
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    with open(out, "w") as fh:
        # Header: n pixel_size lambda cs amp_contrast true_dz
        fh.write(f"{args.n} {args.pixel:.6f} {lam:.6f} {args.cs:.6e} "
                 f"{args.ac:.6f} {args.defocus:.6f}\n")
        # Body: row-major pixels, 6 decimals, wrapped one row per line for sanity.
        for y in range(args.n):
            fh.write(" ".join(f"{img[y, x]:.6f}" for x in range(args.n)) + "\n")

    print(f"[make_synthetic] wrote {out}")
    print(f"[make_synthetic] n={args.n} pixel={args.pixel} A  lambda={lam:.5f} A  "
          f"Cs={args.cs:.3e} A  ac={args.ac}  TRUE defocus={args.defocus:.1f} A (synthetic)")


if __name__ == "__main__":
    main()
