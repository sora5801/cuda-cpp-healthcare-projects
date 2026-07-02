#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic coronary network sample
# ---------------------------------------------------------------------------
# Project 6.20 : Coronary Autoregulation & Microvascular Perfusion
#
# WHAT THIS MAKES (and why it is engineered, not random)
#   A tiny, fully SYNTHETIC coronary microvascular network as a graph of nodes
#   (junctions) and vessel segments. It is deliberately built so the demo tells
#   a clear story (PATTERNS.md §6 "make the sample interpretable"):
#
#     * node 0 is the ARTERIAL INLET, pinned at aortic pressure (Dirichlet).
#     * two capillary-bed OUTLETS are pinned at a low venous pressure (Dirichlet).
#     * interior junction nodes carry the sparse flow-conservation equations.
#     * ONE segment is a STENOSIS: an abnormally narrow vessel on the path to one
#       territory. Because Poiseuille conductance ~ r^4, that narrowing drops the
#       downstream (distal) pressure, so the computed virtual FFR for that branch
#       comes out FLOW-LIMITING (< 0.80) while the healthy branch stays near 1.0.
#     * every segment carries a metabolic TARGET flow the autoregulation loop
#       drives toward by adjusting radii.
#
#   This is study material, NOT patient data and NOT clinically valid. The file
#   format is documented in data/README.md and parsed by src/reference_cpu.cpp.
#
# USAGE
#   python scripts/make_synthetic.py                # writes data/sample/coronary_network.txt
#   python scripts/make_synthetic.py --out foo.txt  # custom path
#
# The generator is DETERMINISTIC (no RNG): the committed sample is reproducible
# byte-for-byte, which keeps demo/expected_output.txt stable.
# ===========================================================================
import argparse
import os

# --- Network topology (indices are node ids) --------------------------------
# A small binary-ish coronary tree with two perfusion territories:
#
#        (0 aorta inlet, 100 mmHg)
#              |  seg0  (epicardial artery, wide)
#        (1 main junction)
#           /         \
#      seg1            seg2  <-- STENOSIS on the branch to node 3
#       /                 \
#  (2 left junc)       (3 right junc, distal to stenosis)
#     |  seg3             |  seg4
#  (4 left bed)        (5 right bed)
#     | seg5              | seg6
#  (6 venous, 20)     (7 venous, 20)   <-- both pinned low (outlets)
#
# Plus a small cross-link (seg7) between the two junctions so the Laplacian is
# not a pure tree (makes the sparse solve non-trivial and the network realistic).
#
# Each segment: (a, b, radius_um, length_um, target_flow). Radii/lengths are in
# the arteriole..small-artery range; targets are order-of-magnitude flows.
NODES = [
    # (is_fixed, pressure_mmHg)
    (1, 100.0),   # 0 aortic inlet
    (0,   0.0),   # 1 main junction
    (0,   0.0),   # 2 left junction
    (0,   0.0),   # 3 right junction (distal to stenosis)
    (0,   0.0),   # 4 left capillary bed
    (0,   0.0),   # 5 right capillary bed
    (1,  20.0),   # 6 venous outlet (left)
    (1,  20.0),   # 7 venous outlet (right)
]

# a, b, radius_um, length_um, target_flow(um^3/s)
# The two territories are deliberately NOT identical: the right branch carries
# the stenosis (seg2, r=8um) and a longer distal arteriole, so its distal
# pressure (node 3) stays visibly BELOW the healthy left branch (node 2) even
# after autoregulation dilates the vessels -- that asymmetry is the whole point.
SEGMENTS = [
    (0, 1, 30.0, 2000.0, 1.0e6),   # seg0 epicardial artery (wide)
    (1, 2, 20.0, 1500.0, 5.0e5),   # seg1 healthy branch to left territory
    (1, 3,  8.0, 1500.0, 5.0e5),   # seg2 STENOSIS (narrow!) to right territory
    (2, 4, 15.0, 1000.0, 3.0e5),   # seg3 arteriole (left)
    (3, 5, 11.0, 1600.0, 3.0e5),   # seg4 arteriole (right, longer/narrower)
    (4, 6, 10.0,  800.0, 2.0e5),   # seg5 capillary->vein (left)
    (5, 7, 10.0,  800.0, 2.0e5),   # seg6 capillary->vein (right)
    (2, 3,  6.0, 1800.0, 5.0e4),   # seg7 thin collateral cross-link (limited flow)
]

HCT = 0.45          # hematocrit fraction (feeds Fahraeus-Lindqvist viscosity)
AORTIC_P = 100.0    # inlet pressure (mmHg), for FFR normalization

# The FFR read-out: measure across the stenosis (seg2), proximal node 1, distal 3.
FFR_SEG, FFR_PROX, FFR_DIST = 2, 1, 3


def write_network(path: str) -> None:
    """Emit the network in the whitespace format documented in data/README.md."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lines = []
    lines.append("# SYNTHETIC coronary microvascular network -- project 6.20")
    lines.append("# NOT patient data, NOT clinically valid. See data/README.md.")
    lines.append("# header: n_nodes n_segs hct aortic_p")
    lines.append(f"{len(NODES)} {len(SEGMENTS)} {HCT} {AORTIC_P}")
    lines.append("# nodes: is_fixed  fixed_pressure_mmHg   (one per node, id 0..N-1)")
    for is_fixed, p in NODES:
        lines.append(f"{is_fixed} {p}")
    lines.append("# segments: a b radius_um length_um target_flow")
    for a, b, r, L, tgt in SEGMENTS:
        lines.append(f"{a} {b} {r} {L} {tgt}")
    lines.append("# FFR readout: ffr_seg ffr_prox ffr_dist")
    lines.append(f"{FFR_SEG} {FFR_PROX} {FFR_DIST}")
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[make_synthetic] wrote {path}: "
          f"{len(NODES)} nodes, {len(SEGMENTS)} segments (stenosis on seg{FFR_SEG})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate synthetic coronary network (6.20)")
    here = os.path.dirname(os.path.abspath(__file__))
    default_out = os.path.join(here, "..", "data", "sample", "coronary_network.txt")
    ap.add_argument("--out", default=default_out, help="output path")
    args = ap.parse_args()
    write_network(os.path.normpath(args.out))
