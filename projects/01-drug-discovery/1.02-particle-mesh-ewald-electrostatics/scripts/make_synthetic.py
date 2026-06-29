#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic charge-system sample
# ---------------------------------------------------------------------------
# Project 1.2 : Particle-Mesh Ewald Electrostatics
#
# WHY THIS EXISTS
#   Real solvated-protein boxes (CHARMM-GUI, MemProtMD, Anton trajectories) are
#   large and/or license/credential gated (see scripts/download_data.*). For the
#   offline demo we generate a TINY, clearly-SYNTHETIC charge system the loader
#   understands. It is engineered so the result is INTERPRETABLE and verifiable:
#   an NaCl-like ionic crystal of alternating +1/-1 charges on a simple-cubic
#   lattice, which is charge-neutral (PME requires net-zero charge) and whose
#   reciprocal energy is well-defined and stable.
#
#   File format (matches load_system in src/reference_cpu.cpp):
#       line 1:  "<n> <box>"
#       n lines: "x y z q"   (coordinates in [0, box), charge q in elementary e)
#
#   The box side is kept small (default 8.0 reduced units) so the auto-chosen FFT
#   grid stays modest (K ~ 16-24) and BOTH the CPU naive-DFT reference and the GPU
#   cuFFT run quickly -- the demo must be snappy.
#
# USAGE
#   python scripts/make_synthetic.py                 # default 2x2x2 = 8 ions
#   python scripts/make_synthetic.py --reps 3        # 3x3x3 = 27 ... (still neutral if even count)
#
# NOTE: everything here is SYNTHETIC and for teaching only -- no clinical meaning.
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "charges_sample.txt"


def build(reps: int, box: float):
    """Build a reps x reps x reps simple-cubic lattice of alternating charges.
    Charge of site (i,j,k) is +1 if (i+j+k) even else -1 (the NaCl pattern).
    The lattice spacing fills the box; sites sit at cell centers so none lands
    exactly on a periodic image boundary."""
    spacing = box / reps
    atoms = []
    qsum = 0
    for i in range(reps):
        for j in range(reps):
            for k in range(reps):
                x = (i + 0.5) * spacing
                y = (j + 0.5) * spacing
                z = (k + 0.5) * spacing
                q = 1 if ((i + j + k) % 2 == 0) else -1
                atoms.append((x, y, z, q))
                qsum += q
    # If the lattice has an odd number of sites the net charge is +/-1; flip the
    # last site's charge to restore exact neutrality (PME demands net zero).
    if qsum != 0:
        x, y, z, q = atoms[-1]
        atoms[-1] = (x, y, z, q - qsum)
    return atoms


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic PME charge sample.")
    ap.add_argument("--reps", type=int, default=2, help="ions per axis (reps^3 total)")
    ap.add_argument("--box", type=float, default=8.0, help="cubic box side (reduced units)")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    atoms = build(args.reps, args.box)
    lines = [f"{len(atoms)} {args.box:g}"]
    for (x, y, z, q) in atoms:
        lines.append(f"{x:.6f} {y:.6f} {z:.6f} {q:.1f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  "
          f"(n={len(atoms)}, box={args.box}, reps={args.reps}; SYNTHETIC NaCl-like lattice)")


if __name__ == "__main__":
    main()
