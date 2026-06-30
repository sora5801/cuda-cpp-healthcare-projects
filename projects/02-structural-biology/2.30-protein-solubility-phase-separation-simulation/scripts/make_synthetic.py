#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic HPS system sample
# ---------------------------------------------------------------------------
# Project 2.30 : Protein Solubility & Phase Separation Simulation
#
# WHY THIS EXISTS
#   The real LLPS datasets (PhaSePro, DisProt, FuzDB) are sequence/annotation
#   databases, not ready-to-run particle configurations, and a true multi-million
#   bead condensate is research-grade. For a DETERMINISTIC, offline teaching demo
#   we instead generate a small, clearly-SYNTHETIC coarse-grained system: a handful
#   of short "IDP" chains of sticky beads placed in a periodic box. With high
#   stickiness (lambda) the chains pull together into a single droplet -- the
#   minimal, visible signature of phase separation. Synthetic data is LABELED
#   synthetic everywhere (CLAUDE.md §8); this is NOT real protein structure.
#
#   The output matches the loader in src/reference_cpu.cpp (see data/README.md):
#     header: n_beads n_chains box sigma epsilon r_cut k_bond r0 mass dt n_steps
#     then n_beads rows of:  x y z vx vy vz lambda
#
#   Reproducibility: a fixed RNG seed makes the file byte-identical every run, so
#   demo/expected_output.txt stays stable.
#
# USAGE
#   python scripts/make_synthetic.py                 # default 4 chains x 10 beads
#   python scripts/make_synthetic.py --chains 6 --len 12 --steps 300
# ===========================================================================
import argparse
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "system.txt"


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic HPS LLPS system.")
    ap.add_argument("--chains", type=int, default=6, help="number of IDP chains")
    ap.add_argument("--len", type=int, default=6, help="beads per chain")
    ap.add_argument("--box", type=float, default=7.0, help="cubic box side (reduced units)")
    ap.add_argument("--sigma", type=float, default=1.0, help="bead diameter (LJ length)")
    ap.add_argument("--epsilon", type=float, default=1.0, help="LJ well depth (energy)")
    ap.add_argument("--rcut", type=float, default=2.5, help="non-bonded cutoff (in sigma units)")
    ap.add_argument("--kbond", type=float, default=50.0, help="harmonic bond stiffness")
    ap.add_argument("--r0", type=float, default=1.0, help="bond rest length")
    ap.add_argument("--mass", type=float, default=1.0, help="bead mass")
    ap.add_argument("--dt", type=float, default=0.002, help="integration time step")
    ap.add_argument("--steps", type=int, default=120, help="number of MD steps")
    ap.add_argument("--lam", type=float, default=0.9, help="stickiness lambda in [0,1]")
    ap.add_argument("--seed", type=int, default=2030, help="RNG seed (determinism)")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    rng = random.Random(args.seed)                     # fixed seed => reproducible file
    n_chains, clen = args.chains, args.len
    n = n_chains * clen
    box = args.box

    # Lay each chain out as a STRAIGHT rod of beads spaced exactly r0 apart along
    # +z, anchored on a coarse 3-D grid of "slots" so that:
    #   * bonded neighbours start at the bond rest length (zero bond force), and
    #   * non-bonded beads start at least ~sigma apart (no Lennard-Jones hard-core
    #     overlap), which is what keeps the integrator STABLE -- a random-walk
    #     start can place two beads on top of each other and the ~r^-12 repulsion
    #     then launches them to infinity (a real MD failure mode; see THEORY §5).
    # The chains start SEPARATED; letting the sticky HPS attraction pull them into
    # one droplet over the run is what makes the condensate a real result, not a
    # planted one. A tiny random jitter breaks the perfect symmetry.
    rows = []
    # Arrange the chains as parallel rods on a 2-D (x,y) grid, each rod running
    # along +z. Each rod occupies its own (x,y) column, so rods never overlap;
    # within a rod beads sit exactly r0 apart (no bond strain), and the rod's
    # z-extent (clen-1)*r0 is assumed to fit in the box. The columns are spaced
    # so neighbouring rods start a few sigma apart -- close enough for the sticky
    # HPS attraction to draw them together, far enough to avoid an initial clash.
    per_side = max(1, math.ceil(math.sqrt(n_chains)))   # columns per box edge
    spacing = box / (per_side + 1)                      # gap between columns
    z_center = 0.5 * box                                # center each rod in z
    az0 = z_center - 0.5 * (clen - 1) * args.r0
    placed = 0
    for gx in range(per_side):
        for gy in range(per_side):
            if placed >= n_chains:
                break
            cx = (gx + 1) * spacing
            cy = (gy + 1) * spacing
            for b in range(clen):
                # straight rod along z + small jitter so the start is not perfectly rigid
                x = (cx + rng.uniform(-0.05, 0.05)) % box
                y = (cy + rng.uniform(-0.05, 0.05)) % box
                z = (az0 + b * args.r0 + rng.uniform(-0.05, 0.05)) % box
                # Small random initial velocities (a gentle "temperature").
                vx = rng.uniform(-0.05, 0.05)
                vy = rng.uniform(-0.05, 0.05)
                vz = rng.uniform(-0.05, 0.05)
                rows.append((x, y, z, vx, vy, vz, args.lam))
            placed += 1

    header = (f"{n} {n_chains} {box:g} {args.sigma:g} {args.epsilon:g} "
              f"{args.rcut:g} {args.kbond:g} {args.r0:g} {args.mass:g} "
              f"{args.dt:g} {args.steps}")

    lines = ["# SYNTHETIC HPS coarse-grained LLPS system (NOT real protein data)",
             "# header: n_beads n_chains box sigma epsilon r_cut k_bond r0 mass dt n_steps",
             header,
             "# rows: x y z vx vy vz lambda"]
    for (x, y, z, vx, vy, vz, lam) in rows:
        lines.append(f"{x:.6f} {y:.6f} {z:.6f} {vx:.6f} {vy:.6f} {vz:.6f} {lam:.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  "
          f"(n={n}, chains={n_chains}x{clen}, steps={args.steps}; SYNTHETIC)")


if __name__ == "__main__":
    main()
