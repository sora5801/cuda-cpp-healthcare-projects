#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic CTC volume sample
# ---------------------------------------------------------------------------
# Project 4.31 : Virtual Colonoscopy & CT Colonography
#
# WHY THIS EXISTS
#   Real CT colonography volumes (TCIA, NLST, ...) are large and patient-derived;
#   we never commit those. Instead we generate a tiny, CLEARLY-SYNTHETIC volume
#   that has the SAME structure a fly-through renderer cares about:
#
#     * a hollow, air-filled colonic LUMEN  -> low density (~0.0)
#     * a soft-tissue WALL around it        -> high density (~1.0)
#     * one small POLYP bump on the wall    -> the "known answer" the demo
#                                              recovers (it brightens under the
#                                              virtual endoscope's headlamp)
#
#   The tube is a vertical cylinder along +z (the fly-through axis) with a gentle
#   sideways bend, so the camera placed at the mouth looks straight down the
#   lumen. Density is a SMOOTH ramp across the wall (not a hard step) so the
#   trilinear iso-surface is well defined and the gradient (the surface normal)
#   is meaningful -- exactly what a real, partial-volume CT looks like.
#
#   Output is the plain-text format reference_cpu.h documents:
#     header : "nx ny nz iso step max_steps width height"
#     body   : nx*ny*nz density floats (x fastest, then y, then z)
#
#   Everything here is deterministic (no RNG) so expected_output.txt is stable.
#   Synthetic data is LABELED synthetic everywhere (CLAUDE.md §8); these are NOT
#   real Hounsfield Units and imply nothing clinical.
#
# USAGE
#   python scripts/make_synthetic.py                 # default tiny sample
#   python scripts/make_synthetic.py --nx 96 --width 256   # larger problem
# ===========================================================================
import argparse
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "colon_volume_sample.txt"


def wall_density(dist_to_wall_center, half_thick):
    """Smooth density profile across the colon wall.

    `dist_to_wall_center` is |radial distance - lumen_radius|: 0 at the wall's
    mid-surface, growing into both the lumen (inside) and tissue (outside). We
    return a smoothstep that is ~0 well inside the lumen (air) and ~1 well into
    the tissue, crossing 0.5 (our render iso-value) right at the wall surface.
    A smooth (rather than binary) wall makes the central-difference gradient --
    and therefore the Phong shading -- well behaved.
    """
    # Signed position is encoded by the caller; here we just shape the ramp.
    t = dist_to_wall_center / half_thick
    if t <= -1.0:
        return 0.0          # deep in the lumen -> air
    if t >= 1.0:
        return 1.0          # deep in tissue -> solid wall
    # smoothstep from -1..1 -> 0..1
    u = (t + 1.0) * 0.5
    return u * u * (3.0 - 2.0 * u)


def build_volume(nx, ny, nz, lumen_r, wall_half):
    """Build the nx*ny*nz density grid as a flat list (x fastest)."""
    vol = [0.0] * (nx * ny * nz)

    # Lumen centerline: a vertical tube (along z) that bends gently in x so the
    # colon is not a boring straight pipe. cx(z), cy fixed at the volume center.
    cy = ny * 0.5
    bend_amp = nx * 0.12                      # how far the tube wanders in x
    cx0 = nx * 0.5

    # Polyp: a spherical bump growing INWARD from the wall (into the lumen) on
    # the -y side of the tube. With the camera's up = +y, that wall projects to
    # the UPPER-CENTER of the frame -- the window main.cu measures. The polyp
    # center is inset slightly INSIDE the lumen radius so the sphere bulges into
    # the air, creating a convex lump the headlamp lights up brightly. Placed
    # partway down the tube (z = 0.45*nz) so the camera sees it clearly. These
    # values are tuned so the polyp window reads ~2x brighter than the smooth
    # wall around it -- a clear, recoverable "known answer" (THEORY §6).
    pz = nz * 0.45                            # polyp center z (down the tube)
    polyp_r = lumen_r * 0.90                  # polyp radius (fraction of lumen)
    polyp_inset = 3.0                         # how far inside the wall it sits

    for k in range(nz):
        cx = cx0 + bend_amp * math.sin(math.pi * k / nz)   # bent centerline x
        # Polyp center in x/y at this slice: on the lower (-y) lumen wall,
        # pulled `polyp_inset` voxels into the lumen so the bump is convex.
        polyp_cx = cx
        polyp_cy = cy - (lumen_r - polyp_inset)
        for j in range(ny):
            for i in range(nx):
                dx = i - cx
                dy = j - cy
                r = math.sqrt(dx * dx + dy * dy)    # radial distance from axis
                # Effective wall position: distance from the lumen surface.
                # Negative inside the lumen, positive in the tissue.
                signed = r - lumen_r
                d = wall_density(signed, wall_half)

                # Carve the polyp: a sphere bulging into the lumen makes that
                # spot read as WALL (high density) where it would otherwise be
                # air, i.e. a bump on the inner surface.
                ddx = i - polyp_cx
                ddy = j - polyp_cy
                ddz = k - pz
                pr = math.sqrt(ddx * ddx + ddy * ddy + ddz * ddz)
                if pr < polyp_r:
                    # Inside the polyp sphere -> tissue. Blend so its surface is
                    # also a smooth iso-crossing (same smoothstep shape).
                    bump = wall_density(polyp_r - pr - wall_half, wall_half)
                    if bump > d:
                        d = bump

                vol[(k * ny + j) * nx + i] = d
    return vol


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic CTC volume sample.")
    ap.add_argument("--nx", type=int, default=32, help="volume x size (voxels)")
    ap.add_argument("--ny", type=int, default=32, help="volume y size (voxels)")
    ap.add_argument("--nz", type=int, default=48, help="volume z size (voxels, fly-through axis)")
    ap.add_argument("--iso", type=float, default=0.5, help="render iso-value (wall surface)")
    ap.add_argument("--step", type=float, default=0.5, help="ray-march step (voxels)")
    ap.add_argument("--max-steps", type=int, default=256, help="max march steps per ray")
    ap.add_argument("--width", type=int, default=48, help="output frame width (px)")
    ap.add_argument("--height", type=int, default=48, help="output frame height (px)")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    nx, ny, nz = args.nx, args.ny, args.nz
    lumen_r = min(nx, ny) * 0.28        # lumen radius (voxels)
    wall_half = 1.5                     # half-thickness of the smooth wall ramp

    vol = build_volume(nx, ny, nz, lumen_r, wall_half)

    # Write header + body. Densities at %.4f keep the file small but exactly
    # reproducible by the C++ loader (which reads them as float).
    lines = [f"{nx} {ny} {nz} {args.iso:g} {args.step:g} {args.max_steps} "
             f"{args.width} {args.height}"]
    # One z-slice per text line keeps the file human-skimmable; the loader reads
    # whitespace-separated floats regardless of line breaks.
    for k in range(nz):
        row = []
        base = k * ny * nx
        for idx in range(ny * nx):
            row.append(f"{vol[base + idx]:.4f}")
        lines.append(" ".join(row))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    n = nx * ny * nz
    print(f"[make_synthetic] wrote {out}")
    print(f"[make_synthetic]   volume {nx}x{ny}x{nz} ({n} voxels), frame "
          f"{args.width}x{args.height}; lumen_r={lumen_r:.1f}  (SYNTHETIC)")


if __name__ == "__main__":
    main()
