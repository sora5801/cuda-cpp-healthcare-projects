#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic Purkinje-tree sample
# ---------------------------------------------------------------------------
# Project 6.17 : Purkinje System & Conduction System Modeling
#
# WHY THIS EXISTS
#   Real Purkinje-network geometries (openCARP / MonoAlg3D_C experiments,
#   NeuroMorpho morphologies, PhysioNet His-bundle electrograms) either require
#   registration or are large research meshes not suited to a tiny committed
#   sample. So we deterministically generate a CLEARLY-SYNTHETIC His-Purkinje
#   tree that exercises the exact loader/format the C++ program reads, and whose
#   result is interpretable (a spread of conduction velocities set by fibre
#   diameter, plus one deliberately slow branch). Synthetic data is LABELED
#   synthetic everywhere (see data/README.md). NOT anatomically calibrated; for
#   teaching only, never clinical use.
#
# FILE FORMAT (matches load_tree() in src/reference_cpu.cpp)
#   line 1 : N  dt_ms  n_steps
#   next N : n_nodes length_mm D stim_amp stim_dur_ms stim_width thresh parent delay_ms
#
#   The values here are kept IN SYNC with build_synthetic_tree() in src/main.cu
#   so the demo output is identical whether the sample file is present or not.
#
# USAGE
#   python scripts/make_synthetic.py                       # default 7-cable tree
#   python scripts/make_synthetic.py --out other_tree.txt
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "purkinje_tree.txt"

# Global integration clock: an explicit forward-Euler step that is stable for the
# diffusion coefficients used below (see THEORY.md "Numerical considerations":
# dt <= dx^2 / (2 D)). 60 ms is enough for the wave to traverse every cable.
DT_MS   = 0.01
N_STEPS = 6000

# One row per cable: (n_nodes, length_mm, D, stim_amp, stim_dur_ms, stim_width,
#                     thresh, parent, delay_ms).
# Every cable is paced at its own proximal (left) end so it always fires; the
# TREE-level timing is assembled afterwards by the graph-delay pass. Diffusion D
# (a proxy for fibre diameter) is what varies the conduction velocity -- the
# teaching point. Cables 2/5/6 are thinner (smaller D) => slower.
CABLES = [
    # His bundle (root): thick/fast, directly paced.
    (65, 20.0, 3.0, 1.0, 2.0, 3, 0.5, -1, 0.0),
    # Left bundle branch: thick/fast.
    (65, 25.0, 3.0, 1.0, 2.0, 3, 0.5,  0, 1.0),
    # Right bundle branch: thinner/slower.
    (65, 25.0, 1.5, 1.0, 2.0, 3, 0.5,  0, 1.0),
    # Terminal Purkinje fascicles (leaves) -> Purkinje-muscle junctions.
    (65, 15.0, 2.5, 1.0, 2.0, 3, 0.5,  1, 0.5),
    (65, 15.0, 2.5, 1.0, 2.0, 3, 0.5,  1, 0.5),
    (65, 15.0, 2.0, 1.0, 2.0, 3, 0.5,  2, 0.5),
    (65, 15.0, 2.0, 1.0, 2.0, 3, 0.5,  2, 0.5),
]


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic Purkinje tree sample.")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    lines = [f"{len(CABLES)} {DT_MS:g} {N_STEPS}"]
    for (n, length, D, amp, dur, width, thr, parent, delay) in CABLES:
        lines.append(f"{n} {length:g} {D:g} {amp:g} {dur:g} {width} {thr:g} {parent} {delay:g}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  ({len(CABLES)} cables; SYNTHETIC, not clinical)")


if __name__ == "__main__":
    main()
