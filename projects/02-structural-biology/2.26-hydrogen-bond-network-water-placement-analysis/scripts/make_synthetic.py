#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic GIST sample dataset
# ---------------------------------------------------------------------------
# Project 2.26 : Hydrogen Bond Network & Water Placement Analysis
#
# WHY THIS EXISTS
#   Real explicit-solvent MD trajectories are large and force-field specific, and
#   the curated GIST benchmark sets (T4 lysozyme, FKBP12) come from external tools
#   (AMBER cpptraj, GISTPP). To keep the demo OFFLINE and reproducible we generate
#   a clearly-SYNTHETIC stand-in that mimics the essential structure GIST analyzes:
#   a small "binding pocket" with a few BURIED, ORDERED water sites plus diffuse
#   bulk-like water, sampled over many frames. Synthetic data is LABELED synthetic
#   everywhere (data/README.md, this header, the printed banner).
#
# WHAT WE ENGINEER (so the result is interpretable, PATTERNS.md §6)
#   * A cubic GIST grid over the pocket (default 10x10x10 voxels @ 0.5 A).
#   * A handful of solute atoms; two carry strong partial charges and act as
#     hydrogen-bond anchors. Waters clustered tightly around the anchors land in
#     the SAME voxel every frame -> high occupancy + favorable energy -> they rank
#     at the TOP of the GIST dG list (the "displace me" sites). This embeds a known
#     answer the program recovers and prints.
#   * Diffuse waters scattered through the box -> low-occupancy bulk-like voxels.
#
# DETERMINISM
#   A fixed RNG seed makes the file byte-identical every run, so the committed
#   sample (and therefore demo/expected_output.txt) never drifts.
#
# OUTPUT FORMAT (matches load_dataset in src/reference_cpu.cpp; see data/README.md)
#   nx ny nz
#   ox oy oz spacing
#   nframes waters_per_frame natoms
#   <natoms lines>:  x y z charge          (solute atoms)
#   <nframes*waters_per_frame lines>: x y z (water oxygens, frame-major)
#
# USAGE
#   python scripts/make_synthetic.py                 # writes data/sample/water_sample.txt
#   python scripts/make_synthetic.py --frames 200    # more frames (sharper stats)
# ===========================================================================
import argparse
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "water_sample.txt"

# --- Grid geometry (a small pocket; keeps the committed sample tiny) ---------
NX = NY = NZ = 10          # 1000 voxels
SPACING = 0.5              # Angstrom voxel edge
OX = OY = OZ = 0.0         # grid minimum corner at the origin
# The grid therefore spans [0, 5] Angstrom in each axis.


def voxel_center(ix, iy, iz):
    """World coordinates of the center of voxel (ix,iy,iz)."""
    return (OX + (ix + 0.5) * SPACING,
            OY + (iy + 0.5) * SPACING,
            OZ + (iz + 0.5) * SPACING)


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic GIST water sample.")
    ap.add_argument("--frames", type=int, default=120, help="number of MD frames")
    ap.add_argument("--seed", type=int, default=20260628, help="RNG seed (determinism)")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # --- The GIST "displaceable water" we want the demo to recover ------------
    # The WaterMap/GIST story: the BEST waters to displace with a ligand are
    # ORDERED, UNHAPPY waters -- ones pinned in a tight pocket (very high
    # occupancy -> a large entropic penalty -TdS) that also make poor, slightly
    # STRAINED contacts (a mildly unfavorable, POSITIVE dE relative to bulk). Both
    # terms push the GIST free energy dG up, so these sites top the ranking.
    #
    # We engineer exactly that: two ordered water SITES, each at the center of a
    # chosen voxel, CAGED by a shell of neutral "wall" atoms placed at near-contact
    # distance. The cage squeezes the water (positive LJ -> strained dE) and holds
    # it in place every frame (high occupancy -> high g -> high -TdS). This is the
    # known answer the program must surface at ranks 1-2.
    CAGE_R = 3.2                # Angstrom, cage atom distance (just inside the LJ min)

    site_a_vox = (4, 5, 5)
    site_b_vox = (6, 4, 4)
    sax, say, saz = voxel_center(*site_a_vox)
    sbx, sby, sbz = voxel_center(*site_b_vox)

    # Build a 6-atom octahedral cage (+/-x, +/-y, +/-z) of neutral atoms around a
    # site center -- a simple, symmetric "tight pocket" that strains the water.
    def cage(cx, cy, cz):
        return [
            (cx + CAGE_R, cy, cz, 0.0), (cx - CAGE_R, cy, cz, 0.0),
            (cx, cy + CAGE_R, cz, 0.0), (cx, cy - CAGE_R, cz, 0.0),
            (cx, cy, cz + CAGE_R, 0.0), (cx, cy, cz - CAGE_R, 0.0),
        ]

    # --- Solute atoms: x y z charge -----------------------------------------
    # Two octahedral cages (the displaceable-water pockets) plus two mildly charged
    # surface atoms elsewhere that anchor a few diffuse waters (bulk-like context).
    atoms = cage(sax, say, saz) + cage(sbx, sby, sbz) + [
        (0.7, 0.7, 0.7,  0.10),
        (4.3, 4.3, 4.3, -0.10),
    ]

    # --- Ordered (caged) water sites: one water in each cage every frame -----
    # Small jitter (sigma 0.05 A) keeps each frame slightly different (as a real
    # trajectory wiggles) without leaving the voxel, so these two voxels accumulate
    # one ordered water EVERY frame -> the highest occupancy in the whole grid.
    ordered_sites = [
        (sax, say, saz, 0.05),  # (center x, y, z, jitter sigma in Angstrom)
        (sbx, sby, sbz, 0.05),
    ]

    n_diffuse = 6               # diffuse bulk-like waters per frame (fill the box)
    waters_per_frame = len(ordered_sites) + n_diffuse

    # Diffuse waters must not OVERLAP a solute atom (a real water cannot sit inside
    # an atom's van der Waals volume). We reject any uniform sample closer than
    # this exclusion radius to any atom, so the diffuse background carries only mild
    # energies and never produces a spurious 1/r^12 spike that would dominate the
    # ranking. (The ordered waters are placed at a favorable H-bond distance, so
    # they are already outside this radius by construction.)
    EXCLUSION = 2.6             # Angstrom, matches the energy model's contact floor

    def too_close(wx, wy, wz):
        for (x, y, z, _q) in atoms:
            if (wx - x) ** 2 + (wy - y) ** 2 + (wz - z) ** 2 < EXCLUSION ** 2:
                return True
        return False

    water_lines = []
    for _ in range(args.frames):
        # Ordered waters: Gaussian jitter around their site center.
        for (cx, cy, cz, sig) in ordered_sites:
            wx = cx + rng.gauss(0.0, sig)
            wy = cy + rng.gauss(0.0, sig)
            wz = cz + rng.gauss(0.0, sig)
            water_lines.append(f"{wx:.4f} {wy:.4f} {wz:.4f}")
        # Diffuse waters: uniform across the box, rejecting solute overlaps, so they
        # spread thinly over many voxels as a bulk-like background.
        placed = 0
        while placed < n_diffuse:
            wx = rng.uniform(OX, OX + NX * SPACING)
            wy = rng.uniform(OY, OY + NY * SPACING)
            wz = rng.uniform(OZ, OZ + NZ * SPACING)
            if too_close(wx, wy, wz):
                continue
            water_lines.append(f"{wx:.4f} {wy:.4f} {wz:.4f}")
            placed += 1

    # --- Assemble the file ---------------------------------------------------
    lines = []
    lines.append(f"{NX} {NY} {NZ}")
    lines.append(f"{OX:g} {OY:g} {OZ:g} {SPACING:g}")
    lines.append(f"{args.frames} {waters_per_frame} {len(atoms)}")
    for (x, y, z, q) in atoms:
        lines.append(f"{x:.4f} {y:.4f} {z:.4f} {q:.4f}")
    lines.extend(water_lines)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {out}")
    print(f"[make_synthetic] SYNTHETIC: {args.frames} frames x {waters_per_frame} waters, "
          f"{len(atoms)} solute atoms, {NX}x{NY}x{NZ} grid @ {SPACING} A")
    print(f"[make_synthetic] ordered sites at voxels {site_a_vox}, {site_b_vox} "
          f"should top the GIST dG ranking.")


if __name__ == "__main__":
    main()
