#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate a synthetic molecule scene to render
# ---------------------------------------------------------------------------
# Project 2.8 : GPU Molecular Visualization & Ray Tracing
#
# WHY SYNTHETIC
#   Real structures come from the RCSB PDB / EMDB (see scripts/download_data.*
#   and data/README.md). To keep the demo OFFLINE, reproducible, and free of any
#   licensing question, we generate a clearly-SYNTHETIC molecule whose shape is
#   recognizable when ray traced: a short alpha-helix-like backbone of "carbon"
#   spheres wound as a 3-D helix, decorated with smaller "side-chain" atoms and a
#   few bright "oxygen" caps. The helix gives the ambient-occlusion shading
#   something interesting to do (deep grooves between turns), so the rendered
#   image clearly shows 3-D structure -- the whole point of AO.
#
#   A fixed layout (no RNG) makes the output byte-for-byte reproducible, so
#   demo/expected_output.txt is stable across runs and machines.
#
# OUTPUT FORMAT (data/README.md):
#   line 1 : "<n_atoms> <width> <height> <ao_samples>"
#   next n : "<x> <y> <z> <radius> <color>"   (Angstrom, Angstrom, Angstrom, A, id)
#   color ids: 0=C grey, 1=O red, 2=N blue, 3=H white, 4=S yellow
#
# USAGE
#   python scripts/make_synthetic.py                     # default helix
#   python scripts/make_synthetic.py --turns 6 --width 320 --height 320
# ===========================================================================
import argparse
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "molecule_sample.scene"

# Van-der-Waals radii (Angstrom) for the CPK ids we use. These are the standard
# values VMD's "VDW" representation draws; we keep them here so the synthetic
# molecule looks dimensionally plausible.
VDW = {0: 1.70, 1: 1.52, 2: 1.55, 3: 1.20, 4: 1.80}   # C, O, N, H, S


def helix_atoms(turns, per_turn, radius, rise):
    """Build a backbone helix of carbon atoms plus small decorations.
    turns    : number of full turns of the helix
    per_turn : backbone atoms per turn (angular resolution)
    radius   : helix radius in Angstrom (distance of backbone from the axis)
    rise     : z increase per backbone atom (Angstrom) -> overall length
    Returns a list of (x, y, z, vdw_radius, color) tuples.
    The geometry is fully deterministic (pure trig), so the scene -- and thus
    the rendered image and its checksum -- is identical every run."""
    atoms = []
    n_back = turns * per_turn
    for i in range(n_back):
        theta = 2.0 * math.pi * i / per_turn        # angle around the helix axis
        z = (i - n_back / 2.0) * rise               # centre the helix on z=0
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        atoms.append((x, y, z, VDW[0], 0))          # backbone carbon

        # Every 3rd backbone atom carries a small outward "side chain" (a
        # nitrogen); every 5th carries a bright oxygen cap. Extra surface detail
        # gives ambient occlusion more crevices to shade.
        if i % 3 == 0:
            ox = (radius + 1.6) * math.cos(theta)
            oy = (radius + 1.6) * math.sin(theta)
            atoms.append((ox, oy, z, VDW[2], 2))    # outward nitrogen
        if i % 5 == 0:
            ox = (radius + 2.6) * math.cos(theta + 0.4)
            oy = (radius + 2.6) * math.sin(theta + 0.4)
            atoms.append((ox, oy, z + 0.5, VDW[1], 1))  # bright oxygen cap
    return atoms


def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic molecule scene.")
    ap.add_argument("--turns", type=int, default=5, help="helix turns")
    ap.add_argument("--per-turn", type=int, default=10, help="backbone atoms per turn")
    ap.add_argument("--radius", type=float, default=4.0, help="helix radius (Angstrom)")
    ap.add_argument("--rise", type=float, default=1.0, help="z rise per atom (Angstrom)")
    ap.add_argument("--width", type=int, default=200, help="image width (pixels)")
    ap.add_argument("--height", type=int, default=200, help="image height (pixels)")
    ap.add_argument("--ao", type=int, default=32, help="ambient-occlusion samples/pixel")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    atoms = helix_atoms(args.turns, args.per_turn, args.radius, args.rise)

    lines = [f"{len(atoms)} {args.width} {args.height} {args.ao}"]
    for (x, y, z, r, c) in atoms:
        # 6 decimals keeps the file compact yet exactly reproducible as float32.
        lines.append(f"{x:.6f} {y:.6f} {z:.6f} {r:.3f} {c}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  "
          f"({len(atoms)} atoms, {args.width}x{args.height}, AO={args.ao}; SYNTHETIC helix)")


if __name__ == "__main__":
    main()
