#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic MD starting configuration
# ---------------------------------------------------------------------------
# Project 1.1 : Molecular Dynamics Engine  (reduced-scope teaching version)
#
# WHY THIS EXISTS
#   Real biomolecular MD inputs (CHARMM36m / AMBER topologies, PDB structures)
#   are large, force-field-specific, and partly license-restricted. For a TINY,
#   offline, reproducible demo we instead generate a clearly-SYNTHETIC starting
#   configuration of a Lennard-Jones fluid: atoms on a simple-cubic lattice in a
#   periodic box, with small deterministic velocities. Synthetic data is always
#   labeled synthetic (CLAUDE.md §8).
#
#   This MUST match the loader in src/reference_cpu.cpp (load_system) AND the
#   built-in fallback make_default_system() so the demo's expected_output.txt is
#   stable. The file layout is:
#       line 1 : n box dt steps eps sigma rcut mass
#       n lines: x y z vx vy vz       (one atom per line)
#   All quantities are in LJ reduced units (mass = eps = sigma = 1).
#
# DETERMINISM
#   Velocities come from the SAME splitmix-style integer hash the C++ fallback
#   uses (no floating-point RNG, no platform-dependent std::rand), so the Python
#   generator and the C++ fallback produce byte-identical atoms.
#
# USAGE
#   python scripts/make_synthetic.py                 # 3x3x3 = 27-atom sample
#   python scripts/make_synthetic.py --side 8        # 512-atom bigger system
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "lj_sample.txt"


def hash32(v: int) -> int:
    """32-bit integer hash; matches the C++ fallback's mixing constants exactly."""
    return (v * 2654435761 + 1013904223) & 0xFFFFFFFF


def unit(v: int) -> float:
    """Map a 32-bit hash to [-0.5, 0.5) deterministically (no float RNG state)."""
    return ((v & 0xFFFF) / 65536.0) - 0.5


def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic LJ MD start config.")
    ap.add_argument("--side", type=int, default=3, help="atoms per box edge (n = side^3)")
    ap.add_argument("--spacing", type=float, default=1.2, help="lattice spacing (> sigma)")
    ap.add_argument("--dt", type=float, default=0.004, help="timestep (reduced units)")
    ap.add_argument("--steps", type=int, default=50, help="number of Verlet steps")
    ap.add_argument("--rcut", type=float, default=2.5, help="LJ cutoff (sigma units)")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    side, spacing = args.side, args.spacing
    n = side ** 3
    box = side * spacing                  # box tiles the lattice (periodic)
    eps = sigma = mass = 1.0              # reduced units

    lines = [f"{n} {box:g} {args.dt:g} {args.steps} {eps:g} {sigma:g} {args.rcut:g} {mass:g}"]
    idx = 0
    for a in range(side):
        for b in range(side):
            for c in range(side):
                x, y, z = a * spacing, b * spacing, c * spacing
                h = hash32(idx)
                # Match the C++ fallback's three independent velocity components.
                vx = 0.1 * unit(h)
                vy = 0.1 * unit((h * 2246822519 + 1) & 0xFFFFFFFF)
                vz = 0.1 * unit((h * 3266489917 + 7) & 0xFFFFFFFF)
                lines.append(f"{x:.6f} {y:.6f} {z:.6f} {vx:.6f} {vy:.6f} {vz:.6f}")
                idx += 1

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  (n={n}, box={box:g}; SYNTHETIC LJ fluid)")


if __name__ == "__main__":
    main()
