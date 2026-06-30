#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Write the MDFF problem (density + atoms)
# ---------------------------------------------------------------------------
# Project 2.12 : Flexible Fitting / MDFF
#
# WHY THIS EXISTS
#   Real cryo-EM maps (EMDB) and structures (PDB) are public but bulky and need
#   parsing (MRC/CCP4, PDB formats). To keep the demo offline and tiny, the "data"
#   here is a clearly-SYNTHETIC fitting problem: a small atomic model that has
#   been misfitted from its target, plus the parameters of a Gaussian density
#   simulated from the target. Synthetic data is always LABELED synthetic.
#
#   To keep the committed sample small we DO NOT store the full nx*ny*nz density
#   grid -- we store the atom positions and the blob width `sigma`, and the
#   program rebuilds the density by summing a Gaussian around each TARGET atom
#   (exactly what the C++ build_density does). The format is parsed by
#   reference_cpu.cpp::load_problem().
#
# OUTPUT (data/README.md format):
#   line 1 : nx ny nz vox w_dens k_rest step iters natoms sigma
#   next natoms lines : x0_x x0_y x0_z     (starting/misfitted positions)
#   next natoms lines : tx  ty  tz         (ground-truth target positions)
#
# The geometry mirrors make_synthetic() in reference_cpu.cpp: a 3x3x3 lattice of
# 27 atoms centred in a 24^3 map, each displaced by a fixed deterministic offset.
# (The program's built-in fallback produces the same numbers; this script lets a
# learner regenerate or resize the committed sample.)
#
# USAGE
#   python scripts/make_synthetic.py                 # writes data/sample/mdff_problem.txt
#   python scripts/make_synthetic.py --iters 400     # longer fit
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "mdff_problem.txt"


def main():
    ap = argparse.ArgumentParser(description="Write the synthetic MDFF problem.")
    ap.add_argument("--n", type=int, default=32, help="density grid size (cubic)")
    ap.add_argument("--vox", type=float, default=1.0, help="voxel size (units/voxel)")
    ap.add_argument("--w_dens", type=float, default=6.0, help="density-force weight")
    ap.add_argument("--k_rest", type=float, default=0.05, help="restraint stiffness")
    ap.add_argument("--step", type=float, default=0.05, help="steepest-descent step")
    ap.add_argument("--iters", type=int, default=200, help="fitting iterations")
    ap.add_argument("--sigma", type=float, default=1.2, help="density blob width")
    ap.add_argument("--spacing", type=float, default=6.0, help="atom lattice spacing")
    ap.add_argument("--dispmag", type=float, default=1.0, help="per-axis misfit displacement")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    n, L = args.n, args.spacing
    # Map centre (matches the C++ builder: 0.5*(n-1)*vox per axis).
    c = 0.5 * (n - 1) * args.vox

    targets, starts = [], []
    a = 0
    for kz in (-1, 0, 1):
        for ky in (-1, 0, 1):
            for kx in (-1, 0, 1):
                tgt = (c + kx * L, c + ky * L, c + kz * L)
                # Fixed deterministic displacement (no RNG) -> reproducible.
                m = args.dispmag
                disp = (m * ((a % 3) - 1),
                        m * (((a // 3) % 3) - 1),
                        m * (((a // 9) % 3) - 1))
                start = (tgt[0] + disp[0], tgt[1] + disp[1], tgt[2] + disp[2])
                targets.append(tgt)
                starts.append(start)
                a += 1

    natoms = len(targets)
    lines = [f"{n} {n} {n} {args.vox:g} {args.w_dens:g} {args.k_rest:g} "
             f"{args.step:g} {args.iters} {natoms} {args.sigma:g}"]
    lines += [f"{x:.6f} {y:.6f} {z:.6f}" for (x, y, z) in starts]
    lines += [f"{x:.6f} {y:.6f} {z:.6f}" for (x, y, z) in targets]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  "
          f"({natoms} atoms, {n}^3 map, sigma={args.sigma}, {args.iters} iters; SYNTHETIC)")


if __name__ == "__main__":
    main()
