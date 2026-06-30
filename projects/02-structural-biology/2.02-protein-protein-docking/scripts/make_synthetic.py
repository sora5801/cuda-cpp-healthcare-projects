#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate a synthetic docking pair (receptor +
#                                ligand) with a KNOWN best translation
# ---------------------------------------------------------------------------
# Project 2.2 : Protein-Protein Docking  (catalog id 2.2)
#
# WHY THIS EXISTS
#   Real complexes (Docking Benchmark 5.5, PDB) need download + parsing and have
#   no single "right" rigid answer to verify against bit-for-bit. For a teaching
#   demo we instead engineer a tiny SYNTHETIC pair whose correct docking
#   translation we KNOW exactly, so the FFT search has a verifiable target
#   (PATTERNS.md section 6 "synthetic data that makes the demo interpretable").
#   Synthetic data is always LABELED synthetic (CLAUDE.md section 8).
#
# THE CONSTRUCTION (so the answer is unambiguous)
#   * RECEPTOR: an irregular "protein blob" -- the union of several overlapping
#     spheres of atoms at different centers/sizes. The irregular, non-symmetric
#     surface gives the shape correlation a single, sharply-defined best fit.
#   * LIGAND: a COPY of the whole receptor, displaced by a known integer vector
#     D (in voxels), so L(x) = R(x - D). Both proteins are voxelized with the
#     SAME core/skin shape rule, so the search computes the autocorrelation of
#     the shape function. By the Cauchy-Schwarz inequality
#       S(t) = sum_x R(x) * L(x - t) = sum_x R(x) * R(x - t - D)  <=  sum_x R(x)^2
#     attains its UNIQUE global maximum exactly at t = -D (where the two copies
#     re-register). So the FFT search is GUARANTEED to recover t = -D, the
#     translation that slides the ligand back onto the receptor -- a clean,
#     verifiable target (the same "embed a known answer" trick 12.01 uses).
#   * We write the expected recovered translation (-D) into the file header so
#     main.cu can report "RECOVERED".
#
#   HONESTY: docking a protein onto a displaced copy of itself is a real task
#   (homodimer / symmetric self-assembly / crystal-packing search), and it
#   exercises the exact FFT-correlation engine of ZDOCK/ClusPro. It is NOT a
#   blind hetero-complex prediction -- see THEORY "Where this sits in the real
#   world" for what production docking adds (rotations, electrostatics, ranking).
#
#   The grid is small (default N=32) so the O(Ng^2) CPU reference finishes in a
#   second or two -- it is a teaching baseline, not a production grid.
#
# OUTPUT (data/README.md format):
#   header:  "n_recv n_lig N spacing  T_x T_y T_z"
#   then n_recv lines "x y z" (receptor atoms, Angstrom)
#   then n_lig  lines "x y z" (ligand   atoms, Angstrom)
#
# USAGE
#   python scripts/make_synthetic.py
#   python scripts/make_synthetic.py --N 48 --spacing 1.5
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "dock_sample.txt"


def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic protein-protein docking pair.")
    ap.add_argument("--N", type=int, default=32, help="grid edge length in voxels")
    ap.add_argument("--spacing", type=float, default=1.5, help="Angstrom per voxel")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    N, sp = args.N, args.spacing
    step = sp                     # atom lattice spacing (~ one atom per voxel)

    # ---- Receptor: an irregular blob = union of overlapping atom spheres ------
    # Centers/radii (Angstrom) chosen to give a lumpy, non-symmetric surface so
    # the shape correlation has a single clear best fit. All in the world frame
    # (voxelize_receptor centers the blob in the grid).
    blobs = [
        ((0.0,  0.0,  0.0), 7.0 * sp),
        ((5.0 * sp,  2.0 * sp, -1.0 * sp), 5.0 * sp),
        ((-4.0 * sp, 4.0 * sp,  2.0 * sp), 4.5 * sp),
        ((1.0 * sp, -5.0 * sp,  4.0 * sp), 4.0 * sp),
    ]
    rng_lo = -10.0 * sp
    rng_hi = 10.0 * sp

    def in_receptor(x, y, z):
        for (cx, cy, cz), r in blobs:
            if (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= r * r:
                return True
        return False

    recv = []
    x = rng_lo
    while x <= rng_hi + 1e-6:
        y = rng_lo
        while y <= rng_hi + 1e-6:
            z = rng_lo
            while z <= rng_hi + 1e-6:
                if in_receptor(x, y, z):
                    recv.append((x, y, z))
                z += step
            y += step
        x += step

    # ---- Ligand: the whole receptor, DISPLACED by D voxels --------------------
    # D is in VOXELS; we move the ligand by D * spacing Angstrom. After voxelizing
    # both with the SAME core/skin rule, the ligand grid equals the receptor grid
    # shifted by +D voxels:  L(x) = R(x - D).
    #
    # WHAT THE SEARCH RECOVERS: the docking score is S(t) = sum_x R(x) L(x - t)
    # = sum_x R(x) R(x - t - D), maximal when t + D = 0, i.e. at t = -D. So the
    # correlation peak is the translation that SLIDES THE LIGAND BACK onto the
    # receptor -- the negative of the displacement we applied. We therefore record
    # the EXPECTED RECOVERED TRANSLATION  answer = -D  in the file header, so
    # main.cu's "RECOVERED" check compares against what the search should report.
    D = (3, 2, -1)                                   # displacement applied (voxels)
    answer = (-D[0], -D[1], -D[2])                   # translation the search finds
    disp = (D[0] * sp, D[1] * sp, D[2] * sp)
    lig = [(ax + disp[0], ay + disp[1], az + disp[2]) for (ax, ay, az) in recv]
    header = f"{len(recv)} {len(lig)} {N} {sp:g}  {answer[0]} {answer[1]} {answer[2]}"
    lines = [header]
    lines += [f"{a:.4f} {b:.4f} {c:.4f}" for (a, b, c) in recv]
    lines += [f"{a:.4f} {b:.4f} {c:.4f}" for (a, b, c) in lig]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  "
          f"(receptor {len(recv)} atoms + ligand {len(lig)} atoms, N={N}, "
          f"spacing={sp:g} A; ligand displaced by D={D}, search should recover "
          f"t={answer}; SYNTHETIC docking pair)")


if __name__ == "__main__":
    main()
