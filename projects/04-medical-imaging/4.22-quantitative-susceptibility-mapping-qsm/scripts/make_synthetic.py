#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic QSM field-map sample
# ---------------------------------------------------------------------------
# Project 4.22 : Quantitative Susceptibility Mapping (QSM)
#
# WHY THIS EXISTS
#   Real QSM datasets (QSM Reconstruction Challenge 2.0, HCP 7T, UK Biobank) are
#   large and/or require registration, so we cannot redistribute them. The
#   committed demo therefore runs on a TINY, clearly-SYNTHETIC field map generated
#   here. Everything is deterministic (no RNG) so demo/expected_output.txt is
#   stable across runs and machines.
#
# WHAT IT BUILDS  (the QSM FORWARD problem)
#   1. A known susceptibility phantom chi(r): a zero background with a few compact
#      "sources" -- three paramagnetic blobs (like iron-rich deep-brain nuclei)
#      and one diamagnetic blob (like a calcification). This is the ground truth
#      the C++ reconstruction must recover; main.cu rebuilds the SAME phantom via
#      make_ground_truth_chi() to score recovery. Keep the two in lockstep.
#   2. The measured field map: apply the DIPOLE forward model in k-space,
#         Fhat_field[k] = D(k) * Fhat_chi[k],   D(k) = 1/3 - kz^2/|k|^2   (B0 || z)
#      i.e. DFT(chi) -> multiply by D(k) -> inverse DFT. This is the IDENTICAL
#      operator as make_field_from_chi() in src/reference_cpu.cpp.
#   3. We write ONLY the field map (what a scanner phase image gives, after
#      unwrapping + background removal). The C++ program never sees chi; it must
#      invert the dipole kernel to recover it.
#
#   Output format (matches load_volume() in src/reference_cpu.cpp):
#       header: "<nx> <ny> <nz>"
#       then nx*ny*nz space-separated floats, x fastest then y then z.
#
#   NOTE ON THE DFT: we use a plain O(N^2) direct DFT here (numpy-free, pure
#   Python) so the script has zero dependencies. It is slow, which is fine for a
#   tiny grid; do not raise the size much without switching to numpy.fft.
#
# USAGE
#   python scripts/make_synthetic.py                    # default 16x16x8 sample
#   python scripts/make_synthetic.py --nx 24 --ny 24 --nz 12
# ===========================================================================
import argparse
import cmath
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "field_map.txt"

TWO_PI = 2.0 * math.pi


def signed_freq(i, n):
    """FFT bin index -> signed frequency (matches signed_freq() in the C++)."""
    return i if i <= n // 2 else i - n


def dipole_kernel(fx, fy, fz):
    """D(k) = 1/3 - kz^2/|k|^2 (B0 || z). Matches dipole_kernel() in qsm_core.h."""
    k2 = fx * fx + fy * fy + fz * fz
    if k2 == 0.0:
        return 0.0
    return (1.0 / 3.0) - (fz * fz) / k2


def make_ground_truth_chi(nx, ny, nz):
    """Known susceptibility phantom. MUST match make_ground_truth_chi() in main.cu:
    same source voxels, same values, zero background."""
    chi = [[[0.0] * nx for _ in range(ny)] for _ in range(nz)]
    sources = [
        (nx // 4,     ny // 2, nz // 2, 1.0),    # paramagnetic blob 1
        (nx // 2,     ny // 2, nz // 2, 0.6),    # paramagnetic blob 2 (center)
        (3 * nx // 4, ny // 2, nz // 2, 0.8),    # paramagnetic blob 3
        (nx // 2,     ny // 4, nz // 2, -0.7),   # diamagnetic blob (calcification)
    ]
    for (x, y, z, v) in sources:
        if 0 <= x < nx and 0 <= y < ny and 0 <= z < nz:
            chi[z][y][x] = v
    return chi


def dft3(vol, nx, ny, nz, sign):
    """Direct 3-D DFT of a complex volume (list[z][y][x] of complex).
    sign=-1 forward, +1 inverse (caller applies the 1/N for inverse)."""
    out = [[[0j] * nx for _ in range(ny)] for _ in range(nz)]
    for kz in range(nz):
        for ky in range(ny):
            for kx in range(nx):
                acc = 0j
                for z in range(nz):
                    pz = TWO_PI * kz * z / nz
                    for y in range(ny):
                        py = TWO_PI * ky * y / ny
                        for x in range(nx):
                            px = TWO_PI * kx * x / nx
                            theta = sign * (px + py + pz)
                            acc += vol[z][y][x] * cmath.exp(1j * theta)
                out[kz][ky][kx] = acc
    return out


def make_field_from_chi(chi, nx, ny, nz):
    """Forward dipole model: DFT(chi) -> * D(k) -> inverse DFT. Real output."""
    cchi = [[[complex(chi[z][y][x], 0.0) for x in range(nx)]
             for y in range(ny)] for z in range(nz)]
    spec = dft3(cchi, nx, ny, nz, sign=-1)          # forward DFT
    for kz in range(nz):
        fz = signed_freq(kz, nz) / nz
        for ky in range(ny):
            fy = signed_freq(ky, ny) / ny
            for kx in range(nx):
                fx = signed_freq(kx, nx) / nx
                spec[kz][ky][kx] *= dipole_kernel(fx, fy, fz)
    inv = dft3(spec, nx, ny, nz, sign=+1)           # inverse DFT
    N = nx * ny * nz
    return [[[inv[z][y][x].real / N for x in range(nx)]
             for y in range(ny)] for z in range(nz)]


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic QSM field map.")
    ap.add_argument("--nx", type=int, default=16, help="grid size along x")
    ap.add_argument("--ny", type=int, default=16, help="grid size along y")
    ap.add_argument("--nz", type=int, default=8,  help="grid size along z")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    nx, ny, nz = args.nx, args.ny, args.nz
    chi = make_ground_truth_chi(nx, ny, nz)
    field = make_field_from_chi(chi, nx, ny, nz)

    lines = [f"{nx} {ny} {nz}"]
    # x fastest, then y, then z -- exactly load_volume()'s expected order.
    for z in range(nz):
        for y in range(ny):
            lines.append(" ".join(f"{field[z][y][x]:.8f}" for x in range(nx)))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  ({nx}x{ny}x{nz} field map from a "
          f"SYNTHETIC susceptibility phantom; ground truth withheld)")


if __name__ == "__main__":
    main()
