#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic MD-trajectory sample
# ---------------------------------------------------------------------------
# Project 2.17 : Allosteric Network Analysis
#
# WHY THIS EXISTS
#   Real allosteric MD trajectories (GPCRmd, ASD) are large and license-bound, so
#   we ship a TINY, clearly-SYNTHETIC stand-in that lets demo/run_demo run offline
#   with zero downloads. The data is engineered so the result is INTERPRETABLE and
#   VERIFIABLE: we plant a known allosteric coupling and check the analysis
#   recovers it (PATTERNS.md section 6 "synthetic data that makes the demo
#   interpretable"). Synthetic data is LABELED synthetic everywhere it appears.
#
# THE TOY MODEL (see THEORY.md "The science")
#   * N residues placed along an extended backbone (a single chain). The contact
#     graph is therefore a "path graph": each residue contacts only its sequence
#     neighbors, so the only way for a signal to travel from one end to the other
#     is to hop residue-by-residue down the chain -- giving a long, clearly
#     multi-residue allosteric pathway to recover.
#   * The chain is split into three breathing "domains": an allosteric domain (0),
#     a middle hinge domain (1), and the active domain (2). The allosteric site
#     sits near one end (domain 0) and the active site near the FAR end (domain 2),
#     so they are distant in BOTH sequence and space.
#   * A planted ALLOSTERIC MODE: one global collective coordinate drives ALL three
#     domains IN PHASE along the same axis, so every backbone edge is strongly
#     correlated and the shortest -log|C| path threads the whole chain end to end.
#   This produces a DCC matrix with the block + pathway structure real allosteric
#   proteins show, while staying obviously synthetic and reproducible (fixed seed).
#
#   Output format (parsed by src/reference_cpu.cpp::load_trajectory):
#     # comment lines (incl. "# SITE_ALLO i" and "# SITE_ACTIVE j" annotations)
#     N T                         <- dimensions: residues, frames
#     x y z                       <- N lines per frame, T frames (frame-major)
#     ...
#
# USAGE
#   python scripts/make_synthetic.py                 # default tiny sample
#   python scripts/make_synthetic.py --residues 48 --frames 200
#
# NOTE: standard library only (random.Random with a fixed seed) so the committed
#   sample is byte-reproducible on any machine without numpy.
# ===========================================================================
import argparse
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "trajectory.txt"


def build_equilibrium(n):
    """Place n residues along an extended single-chain backbone.

    Returns a list of (x, y, z) equilibrium positions on a straight chain spaced
    at ~3.8 A (the Cα-Cα virtual bond length). With an 8 A contact cutoff, each
    residue contacts ONLY its sequence neighbors, so the contact graph is a path
    graph: a signal must walk the chain residue-by-residue. That makes the
    recovered allosteric pathway long and unambiguous. Units are arbitrary
    'angstrom-like' to keep the demo dimensionless.
    """
    # 5.0 A spacing with the 8 A contact cutoff means a residue touches its
    # IMMEDIATE neighbors (5 A < 8) but not its next-nearest (10 A > 8): the
    # contact graph is a clean path graph, so every residue lies on the pathway.
    spacing = 5.0
    return [[i * spacing, 0.0, 0.0] for i in range(n)]


def domain_of(i, n):
    """Assign residue i to one of three breathing domains (0,1,2).

    The protein has an allosteric domain (0), a hinge/middle domain (1), and the
    active domain (2). The active site sits in domain 2 and the allosteric site in
    domain 0, so a real cross-domain pathway must exist for them to communicate.
    """
    third = n // 3
    if i < third:
        return 0
    if i < 2 * third:
        return 1
    return 2


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic allosteric MD trajectory.")
    ap.add_argument("--residues", type=int, default=30, help="number of Cα residues")
    ap.add_argument("--frames", type=int, default=120, help="number of trajectory frames")
    ap.add_argument("--seed", type=int, default=20260628, help="RNG seed (reproducible)")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    n, T = args.residues, args.frames
    rng = random.Random(args.seed)      # fixed seed -> byte-identical sample

    eq = build_equilibrium(n)

    # The two annotated functional sites: one near each end of the chain so they
    # are distant in both sequence and space (the hallmark of allostery).
    site_allo = 2                       # inside the allosteric domain (0)
    site_active = n - 3                 # inside the active domain (2)

    # How strongly each domain couples to the GLOBAL allosteric coordinate s(t).
    # Domains 0 and 2 (the two functional ends) couple strongly and IN PHASE, so
    # their motions are highly correlated. The middle hinge domain (1) couples
    # only partially and carries extra independent motion of its own -- this makes
    # the domain-boundary edges the WEAKEST links, so the recovered bottleneck hop
    # lands at the hinge, exactly where a real allosteric "hotspot" residue sits.
    domain_coupling = [1.0, 0.55, 1.0]  # fraction of s(t) each domain follows
    domain_amp = [1.4, 1.2, 1.4]        # overall breathing amplitude per domain

    frames = []
    for t in range(T):
        # ONE global allosteric collective coordinate s(t): a slow sinusoid plus a
        # little noise. Because domains 0,1,2 all couple to s(t) along +z, their
        # motions are correlated -> the planted allosteric communication.
        s = math.sin(2.0 * math.pi * t / T) + 0.15 * rng.gauss(0.0, 1.0)
        # The hinge domain's OWN collective wiggle (independent of s), at a
        # different frequency, that partly decorrelates it from the two ends.
        h = math.cos(2.0 * math.pi * 3.0 * t / T)

        frame = []
        for i in range(n):
            d = domain_of(i, n)
            amp = domain_amp[d]
            # Collective displacement: the shared allosteric mode (weighted by this
            # domain's coupling) plus, for the hinge, its own independent mode.
            drive = domain_coupling[d] * s + (0.8 * h if d == 1 else 0.0)
            cx = 0.0
            cy = 0.0
            cz = amp * drive            # all domains move along +z -> correlated ends
            # Small independent thermal jitter per residue (decorrelates neighbors
            # a little, so off-pathway correlations are weak -- realistic noise).
            jx = 0.25 * rng.gauss(0.0, 1.0)
            jy = 0.25 * rng.gauss(0.0, 1.0)
            jz = 0.25 * rng.gauss(0.0, 1.0)
            frame.append((eq[i][0] + cx + jx,
                          eq[i][1] + cy + jy,
                          eq[i][2] + cz + jz))
        frames.append(frame)

    # ---- Write the file -----------------------------------------------------
    lines = [
        "# SYNTHETIC allosteric MD trajectory (Project 2.17) -- NOT real data",
        "# Generated by scripts/make_synthetic.py with a fixed seed; reproducible.",
        "# Extended single chain; three breathing domains coupled by a shared",
        "# collective mode -> a planted allosteric pathway from SITE_ALLO to SITE_ACTIVE.",
        f"# SITE_ALLO {site_allo}",
        f"# SITE_ACTIVE {site_active}",
        f"{n} {T}",
    ]
    for frame in frames:
        for (x, y, z) in frame:
            lines.append(f"{x:.4f} {y:.4f} {z:.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  "
          f"(N={n}, T={T}, allo={site_allo}, active={site_active}; SYNTHETIC)")


if __name__ == "__main__":
    main()
