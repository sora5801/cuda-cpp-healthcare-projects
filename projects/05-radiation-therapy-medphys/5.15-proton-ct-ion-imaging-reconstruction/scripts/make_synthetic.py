#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate a synthetic proton-CT list-mode set
# ---------------------------------------------------------------------------
# Project 5.15 : Proton CT & Ion Imaging Reconstruction
#
# WHY THIS EXISTS
#   Real proton-CT data (PRaVDA, PRIMA, TOPAS/GATE Monte-Carlo simulations) is
#   large and/or license/registration gated. To let the demo RUN offline with a
#   VERIFIABLE answer, we deterministically synthesise a tiny list-mode dataset
#   from a KNOWN phantom of relative stopping powers (RSP). The data is always
#   LABELED synthetic (data/README.md). No randomness -> byte-stable output.
#
# THE PHANTOM (ground truth we hope to recover)
#   A disc of "water" (RSP=1.0) with two inserts: a denser "bone-like" disc
#   (RSP=1.6) and a lighter "lung-like" disc (RSP=0.3). RSP is dimensionless
#   (relative to water). Coordinates are centimetres.
#
# HOW EACH PROTON'S WEPL IS COMPUTED
#   We emulate a pCT scan: for each projection angle theta, a parallel set of
#   protons enters along direction (cos,sin) at lateral offsets across the field.
#   Each proton's TRUE water-equivalent path length is the line integral of RSP
#   along its MOST-LIKELY PATH (MLP). Crucially we integrate using the SAME MLP
#   (cubic Hermite) and the SAME nearest-voxel sampling the C++ reconstructor
#   uses -- so the generated WEPL is exactly consistent with the forward model,
#   and SART recovers the phantom (docs/PATTERNS.md section 6, "embed a known
#   answer"). Small entry/exit scattering angles are assigned deterministically
#   from a Highland-style estimate so the MLP actually curves (teaching the point
#   that pCT paths are not straight); they are small enough that the straight and
#   curved integrals differ only slightly.
#
# OUTPUT (data/README.md format):
#   header: "<n> <half> <iters> <relax> <path_samples> <n_protons>"
#   then n*n ground-truth RSP floats (row-major),
#   then n_protons rows: "x0 y0 x1 y1 entry_angle exit_angle wepl".
#
# USAGE
#   python scripts/make_synthetic.py
#   python scripts/make_synthetic.py --n 48 --angles 60 --rays 48
# ===========================================================================
import argparse
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "protons_sample.txt"

# Phantom discs: (cx, cy, radius, RSP). World units = cm; RSP relative to water.
PHANTOM = [
    (0.0,  0.0, 4.0, 1.0),    # water background disc
    (1.5,  0.8, 1.2, 1.6),    # dense "bone-like" insert (RSP 1.6)
    (-1.6, -0.9, 1.0, 0.3),   # light "lung-like" insert (RSP 0.3)
]


def rsp_truth_grid(n, half):
    """Rasterise the phantom onto an n x n grid (row-major). Nearest-cell RSP:
    later discs overwrite earlier ones where they overlap (inserts win)."""
    vs = (2.0 * half / (n - 1)) if n > 1 else 0.0
    grid = [0.0] * (n * n)
    for iy in range(n):
        wy = -half + iy * vs
        for ix in range(n):
            wx = -half + ix * vs
            val = 0.0
            for (cx, cy, r, rsp) in PHANTOM:
                if (wx - cx) ** 2 + (wy - cy) ** 2 <= r * r:
                    val = rsp                       # inserts listed later win
            grid[iy * n + ix] = val
    return grid, vs


def world_to_voxel(n, half, vs, x, y):
    """Nearest-voxel index or -1 -- MUST match reference_cpu.cpp exactly."""
    if vs <= 0.0:
        return -1
    ix = math.floor((x + half) / vs + 0.5)
    iy = math.floor((y + half) / vs + 0.5)
    if ix < 0 or ix >= n or iy < 0 or iy >= n:
        return -1
    return iy * n + ix


def mlp_point(x0, y0, x1, y1, a0, a1, t):
    """Cubic-Hermite most-likely path -- MUST match pct_physics.h mlp_point."""
    dx, dy = x1 - x0, y1 - y0
    L = math.sqrt(dx * dx + dy * dy)
    cx, cy = x0 + t * dx, y0 + t * dy
    if L <= 0.0:
        return cx, cy
    vx, vy = -dy / L, dx / L
    m0, m1 = math.tan(a0) * L, math.tan(a1) * L
    t2, t3 = t * t, t * t * t
    h = (t3 - 2.0 * t2 + t) * m0 + (t3 - t2) * m1
    return cx + h * vx, cy + h * vy


def integrate_wepl(grid, n, half, vs, x0, y0, x1, y1, a0, a1, path_samples):
    """True WEPL = integral of RSP along the MLP, sampled the SAME way the
    reconstructor samples it (midpoint quadrature, nearest voxel)."""
    dx, dy = x1 - x0, y1 - y0
    chord = math.sqrt(dx * dx + dy * dy)
    seg = chord / path_samples
    w = 0.0
    for s in range(path_samples):
        t = (s + 0.5) / path_samples
        px, py = mlp_point(x0, y0, x1, y1, a0, a1, t)
        v = world_to_voxel(n, half, vs, px, py)
        if v >= 0:
            w += grid[v] * seg
    return w


def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic proton-CT list-mode dataset.")
    ap.add_argument("--n", type=int, default=32, help="reconstruction grid side (voxels)")
    ap.add_argument("--half", type=float, default=5.0, help="image half-width (cm); spans [-half,half]^2")
    ap.add_argument("--angles", type=int, default=45, help="projection angles over [0,pi)")
    ap.add_argument("--rays", type=int, default=32, help="parallel protons per angle")
    ap.add_argument("--iters", type=int, default=40, help="SART sweeps the solver should run")
    ap.add_argument("--relax", type=float, default=0.80, help="SART relaxation factor")
    ap.add_argument("--path-samples", type=int, default=64, help="MLP quadrature samples per proton")
    ap.add_argument("--scatter-mrad", type=float, default=6.0,
                    help="peak entry/exit scattering angle magnitude (milliradians)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    n, half = args.n, args.half
    grid, vs = rsp_truth_grid(n, half)

    # Protons enter/exit on a line just outside the image on each side. For angle
    # theta the beam direction is u=(cos,sin); protons are offset laterally along
    # v=(-sin,cos). Entry/exit points are +-R along u from the field centre.
    R = 1.6 * half                       # entry/exit stand-off distance (cm)
    scatter = args.scatter_mrad * 1.0e-3  # peak angle in radians

    rows = []
    n_protons = 0
    for ai in range(args.angles):
        theta = math.pi * ai / args.angles
        ux, uy = math.cos(theta), math.sin(theta)
        vx, vy = -math.sin(theta), math.cos(theta)
        for ri in range(args.rays):
            # Lateral offset in [-half, half], evenly spaced.
            frac = (ri / (args.rays - 1)) if args.rays > 1 else 0.5
            off = -half + frac * (2.0 * half)
            x0 = -R * ux + off * vx
            y0 = -R * uy + off * vy
            x1 = R * ux + off * vx
            y1 = R * uy + off * vy
            # Deterministic small scattering angles (rel. to chord). We modulate
            # by the lateral offset so central rays (through more material) bend
            # a touch more -- a crude Highland dependence, purely for teaching.
            depth_frac = 1.0 - min(1.0, abs(off) / half)
            a0 = scatter * depth_frac * math.sin(3.0 * frac + 0.5)
            a1 = -scatter * depth_frac * math.sin(3.0 * frac + 1.1)
            wepl = integrate_wepl(grid, n, half, vs, x0, y0, x1, y1, a0, a1, args.path_samples)
            rows.append(f"{x0:.6f} {y0:.6f} {x1:.6f} {y1:.6f} {a0:.6e} {a1:.6e} {wepl:.6f}")
            n_protons += 1

    header = f"{n} {half:g} {args.iters} {args.relax:g} {args.path_samples} {n_protons}"
    truth_lines = []
    for iy in range(n):
        truth_lines.append(" ".join(f"{grid[iy * n + ix]:.4f}" for ix in range(n)))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        header + "\n" + "\n".join(truth_lines) + "\n" + "\n".join(rows) + "\n",
        encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  "
          f"({n_protons} protons over {args.angles} angles x {args.rays} rays, "
          f"{n}x{n} grid; SYNTHETIC phantom)")


if __name__ == "__main__":
    main()
