#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic trajectory sample
# ---------------------------------------------------------------------------
# Project 1.30 : Trajectory RMSD, Clustering & Contact Analysis
#
# WHY THIS EXISTS
#   Real MD trajectories (MDCATH, GPCRmd, MDDB; see data/README.md) are large
#   and/or need credentials, but the demo must RUN OFFLINE with zero downloads.
#   So we deterministically generate a clearly-SYNTHETIC trajectory that matches
#   the loader's text format and -- crucially -- has a KNOWN, INTERPRETABLE
#   answer (PATTERNS.md sec 6): frame 0 is a compact helix (the reference);
#   later frames progressively "unfold", so the optimal-superposition RMSD grows
#   monotonically and the fraction of native contacts Q decays from 1.0. The
#   demo recovers exactly that shape, which is how a learner sees the analysis
#   "working".
#
#   The data is SYNTHETIC and carries NO physical/biological meaning beyond the
#   geometry we impose. It is labeled synthetic everywhere (CLAUDE.md sec 8).
#
# DETERMINISM
#   We write coordinates with Python's repr() (full round-trip precision) so the
#   C++ loader parses the EXACT same doubles every time -> the program's stdout
#   is byte-identical run to run (required by demo/run_demo).
#
# USAGE
#   python scripts/make_synthetic.py                 # default 12 frames, 16 atoms
#   python scripts/make_synthetic.py --frames 5000   # a bigger trajectory
# ===========================================================================
import argparse
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent           # the project folder
OUT = ROOT / "data" / "sample" / "trajectory_sample.txt"

# Must match N_ATOMS in src/rmsd_core.h. The C++ loader rejects a mismatch.
N_ATOMS = 16


def reference_helix():
    """The compact reference structure (frame 0): N_ATOMS atoms on a tight helix.
    A helix puts non-sequential atoms close together, giving a rich NATIVE
    CONTACT set whose decay we can watch as the structure unfolds.

    Geometry (arbitrary length units, ~Angstrom-like):
      radius 4.0, rise 1.5 per atom, ~100 deg turn per atom.
    Returns a list of (x, y, z) tuples.
    """
    radius = 4.0
    rise = 1.5
    dtheta = math.radians(100.0)
    pts = []
    for i in range(N_ATOMS):
        theta = i * dtheta
        pts.append((radius * math.cos(theta),
                    radius * math.sin(theta),
                    i * rise))
    return pts


def unfold(points, t):
    """Return a copy of `points` partially 'unfolded' by progress t in [0, 1].

    We linearly interpolate each atom toward an EXTENDED straight line along z
    (a fully unfolded chain). At t=0 we get the helix exactly (RMSD 0, Q=1); at
    t=1 we get a straight rod (large RMSD, few native contacts). This is a
    purely geometric morph -- deterministic and monotone in t, so the per-frame
    RMSD increases and Q decreases smoothly. No randomness, so the file (and the
    program output) is fully reproducible.
    """
    extended_rise = 3.8   # straight-chain spacing along z (atoms far apart)
    out = []
    for i, (x, y, z) in enumerate(points):
        ex, ey, ez = 0.0, 0.0, i * extended_rise   # target straight-line atom i
        out.append((x + (ex - x) * t,
                    y + (ey - y) * t,
                    z + (ez - z) * t))
    return out


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic trajectory sample.")
    ap.add_argument("--frames", type=int, default=12, help="number of frames (>= 1)")
    ap.add_argument("--ref", type=int, default=0, help="reference frame index")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    frames = max(1, args.frames)
    ref = args.ref

    base = reference_helix()

    # CONFORMATIONAL STATES, not a smooth ramp. Real trajectories dwell in a few
    # metastable states and hop between them; that is exactly what makes
    # CLUSTERING interesting (many frames collapse onto a few RMSD shells). We
    # therefore drive the unfolding `t` with a 3-state schedule:
    #   - a FOLDED basin near the helix  (small t -> RMSD ~ 0-1, Q ~ 1),
    #   - an INTERMEDIATE basin          (mid t   -> RMSD ~ 4-5, Q partial),
    #   - an UNFOLDED basin              (large t -> RMSD ~ 9-10, Q ~ 0),
    # with a small per-frame jitter so frames within a state are distinct but
    # still land in the same RMSD shell. This yields a multi-peak cluster
    # histogram the demo can show off. The schedule is FIXED (no RNG) so the file
    # is fully reproducible. For frames != 12 we fall back to a smooth ramp.
    if frames == 12:
        # 4 frames per state; tiny jitter keeps them inside one unit-wide shell.
        schedule = [0.00, 0.04, 0.08, 0.10,        # folded   -> RMSD < 1
                    0.42, 0.45, 0.47, 0.49,        # interm.  -> RMSD ~ 4-5
                    0.86, 0.89, 0.92, 0.95]        # unfolded -> RMSD ~ 9-10
    else:
        schedule = [(f / (frames - 1)) if frames > 1 else 0.0 for f in range(frames)]

    lines = [f"{frames} {N_ATOMS} {ref}"]
    for f in range(frames):
        t = schedule[f]
        pts = unfold(base, t)
        for (x, y, z) in pts:
            # repr() preserves full double precision for exact round-trip.
            lines.append(f"{repr(x)} {repr(y)} {repr(z)}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  "
          f"({frames} frames x {N_ATOMS} atoms, ref=frame {ref}; SYNTHETIC)")


if __name__ == "__main__":
    main()
