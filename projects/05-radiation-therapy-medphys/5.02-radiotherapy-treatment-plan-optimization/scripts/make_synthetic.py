#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic FMO plan sample
# ---------------------------------------------------------------------------
# Project 5.2 -- Radiotherapy Treatment-Plan Optimization
#
# WHY THIS EXISTS
#   Real dose-influence ("dij") matrices come from a Monte-Carlo or
#   pencil-beam dose engine on a patient CT and are large + license-encumbered
#   (see data/README.md for the real datasets: OpenKBP, matRad, etc.). To keep
#   the demo runnable OFFLINE we generate a small, clearly-SYNTHETIC 1-D
#   "phantom" FMO instance whose optimum is easy to reason about.
#
# THE SYNTHETIC GEOMETRY (a 1-D slice through a patient)
#   * `n_vox` voxels laid out on a line, index 0..n_vox-1.
#   * A central PTV (tumor) band, an OAR (organ) band just to one side, and the
#     rest is BODY.
#   * `n_beam` beamlets. Beamlet j is "aimed" at voxel center mu_j spread evenly
#     across the phantom; it deposits dose into nearby voxels with a Gaussian
#     falloff D[v,j] = A * exp(-(v - mu_j)^2 / (2 sigma^2)), truncated to a
#     window (that truncation is what makes D SPARSE -- only ~2*window+1
#     nonzeros per beamlet). We emit the matrix in CSR (row = voxel).
#   Because each beamlet's corridor overlaps the PTV differently, the optimizer
#   has real freedom: it turns UP beamlets that hit the PTV and turns DOWN ones
#   that spill into the OAR -- exactly the trade-off inverse planning solves.
#
# DETERMINISM
#   Everything is a fixed analytic function of the indices (no RNG), so the file
#   -- and therefore the demo's stdout -- is byte-identical on every machine.
#   Synthetic data is LABELED synthetic here, in data/README.md, and in stdout.
#
# FILE FORMAT (whitespace-separated tokens; newlines cosmetic). Parsed by
# load_problem() in src/reference_cpu.cpp:
#     n_vox n_beam nnz iters step d_rx
#     <kind target weight>            x n_vox     (kind: 0=PTV 1=OAR 2=BODY)
#     row_ptr[0..n_vox]               (n_vox+1 ints, CSR row pointers)
#     col_idx[0..nnz-1]               (nnz ints, beamlet index of each nonzero)
#     values[0..nnz-1]                (nnz floats, D[v,j] in Gy per unit fluence)
#
# USAGE
#   python scripts/make_synthetic.py                 # writes the tiny sample
#   python scripts/make_synthetic.py --n-vox 4096 --n-beam 256   # bigger problem
# ===========================================================================
import argparse
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent           # the project folder
OUT = ROOT / "data" / "sample" / "plan_sample.txt"

# Structure kind codes -- must match enum StructKind in src/fmo.h.
PTV, OAR, BODY = 0, 1, 2


def build(n_vox, n_beam, sigma, window, amp, iters, step, d_rx,
          ptv_lo, ptv_hi, oar_lo, oar_hi,
          w_ptv, w_oar, w_body, d_oar_max):
    """Return (voxels, row_ptr, col_idx, values) for the synthetic phantom.

    voxels : list of (kind, target, weight)
    CSR    : row_ptr/col_idx/values describing the sparse dose-influence matrix.
    """
    # --- Per-voxel objective specs (the three structures) -------------------
    voxels = []
    for v in range(n_vox):
        if ptv_lo <= v < ptv_hi:
            voxels.append((PTV, d_rx, w_ptv))          # tumor: drive to Rx
        elif oar_lo <= v < oar_hi:
            voxels.append((OAR, d_oar_max, w_oar))     # organ: cap at tolerance
        else:
            voxels.append((BODY, 0.0, w_body))         # rest: keep dose low

    # --- Sparse dose-influence matrix D in CSR (row = voxel) ----------------
    # Beamlet j is centered at mu_j spread across the phantom. For each voxel we
    # collect the beamlets whose Gaussian corridor reaches it (|v - mu_j| <=
    # window); that truncation is what makes the matrix sparse.
    if n_beam > 1:
        mus = [ (n_vox - 1) * j / (n_beam - 1) for j in range(n_beam) ]
    else:
        mus = [ (n_vox - 1) / 2.0 ]

    row_ptr = [0]
    col_idx = []
    values = []
    inv_two_sig2 = 1.0 / (2.0 * sigma * sigma)
    for v in range(n_vox):
        row_nnz = 0
        for j in range(n_beam):
            dv = v - mus[j]
            if abs(dv) <= window:                      # inside this beamlet's window
                w = amp * math.exp(-(dv * dv) * inv_two_sig2)
                # Round to a few decimals so the file (and thus float parsing) is
                # stable and compact; keep only meaningful contributions.
                w = round(w, 6)
                if w > 0.0:
                    col_idx.append(j)
                    values.append(w)
                    row_nnz += 1
        row_ptr.append(row_ptr[-1] + row_nnz)

    return voxels, row_ptr, col_idx, values


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic FMO plan sample.")
    # Defaults define the TINY committed sample: small but non-degenerate, with a
    # PTV, an OAR, real beamlet overlap, and enough iterations to converge.
    ap.add_argument("--n-vox",  type=int,   default=48,  help="number of voxels")
    ap.add_argument("--n-beam", type=int,   default=16,  help="number of beamlets")
    ap.add_argument("--sigma",  type=float, default=2.5, help="beam Gaussian width (voxels)")
    ap.add_argument("--window", type=int,   default=6,   help="truncation half-width (sparsity)")
    ap.add_argument("--amp",    type=float, default=1.0, help="peak dose per unit fluence (Gy)")
    ap.add_argument("--iters",  type=int,   default=400, help="projected-gradient iterations")
    ap.add_argument("--step",   type=float, default=0.02, help="gradient step size eta")
    ap.add_argument("--d-rx",   type=float, default=60.0, help="PTV prescription dose (Gy)")
    ap.add_argument("--out",    default=str(OUT), help="output path")
    args = ap.parse_args()

    nv = args.n_vox
    # Layout the structures: PTV band in the middle third; OAR band just left of it.
    ptv_lo, ptv_hi = int(nv * 0.42), int(nv * 0.58)
    oar_lo, oar_hi = int(nv * 0.25), int(nv * 0.38)

    voxels, row_ptr, col_idx, values = build(
        n_vox=nv, n_beam=args.n_beam, sigma=args.sigma, window=args.window,
        amp=args.amp, iters=args.iters, step=args.step, d_rx=args.d_rx,
        ptv_lo=ptv_lo, ptv_hi=ptv_hi, oar_lo=oar_lo, oar_hi=oar_hi,
        # Weights encode the clinical trade-off: PTV coverage matters most, OAR
        # sparing next, a small BODY pressure keeps stray dose (and the problem)
        # bounded. d_oar_max is the organ tolerance (Gy).
        w_ptv=1.0, w_oar=0.6, w_body=0.02, d_oar_max=25.0)

    nnz = row_ptr[-1]
    lines = [f"{nv} {args.n_beam} {nnz} {args.iters} {args.step:g} {args.d_rx:g}"]
    for (kind, target, weight) in voxels:
        lines.append(f"{kind} {target:g} {weight:g}")
    lines.append(" ".join(str(r) for r in row_ptr))
    lines.append(" ".join(str(c) for c in col_idx))
    lines.append(" ".join(f"{v:g}" for v in values))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  "
          f"(n_vox={nv}, n_beam={args.n_beam}, nnz={nnz}; SYNTHETIC)")


if __name__ == "__main__":
    main()
