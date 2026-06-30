#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic pangenome sample
# ---------------------------------------------------------------------------
# Project 3.30 : Pangenome Graph Construction
#
# WHY THIS EXISTS
#   Real pangenome graphs (HPRC's 94-haplotype human graph, etc.) are large and
#   require building with PGGB/ODGI from licensed assemblies (see download_data.*).
#   For an offline, deterministic, *interpretable* demo we generate a TINY,
#   clearly-SYNTHETIC pangenome by hand here. Synthetic data is labeled synthetic
#   everywhere it appears (CLAUDE.md section 8).
#
# WHAT THE SAMPLE GRAPH MODELS  (a textbook variation graph with "bubbles")
#   A pangenome graph stores several genomes as one sequence graph: NODES are
#   shared sequence segments; each genome is a PATH (a walk) through the nodes.
#   Where genomes agree they share nodes; where they differ the graph "bubbles".
#   We build a backbone the reference walks straight through, plus three variant
#   haplotypes that introduce the three canonical variant types:
#       * SNP / substitution : take an alternate single node instead of the ref one
#       * insertion          : visit an extra node the reference skips
#       * deletion           : skip a node the reference visits
#   These bubbles are exactly what the 1-D layout must untangle so that
#   genomically co-linear nodes end up adjacent on the axis.
#
# OUTPUT FORMAT  (data/README.md describes it; '#' comments are allowed)
#   Line 1            : "N P"          -- N nodes, P paths
#   Line 2            : N node lengths -- in base pairs
#   Next P lines      : "L id0 id1 ..."-- each path's length then its node ids
#
# USAGE
#   python scripts/make_synthetic.py                 # writes data/sample/pangenome_sample.txt
#   python scripts/make_synthetic.py --out other.txt
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "pangenome_sample.txt"

# --- The hand-built synthetic pangenome -------------------------------------
# 12 nodes (0..11). Lengths are "round" base-pair sizes so the printed
# coordinates are easy to read. Nodes 0..9 form a left-to-right backbone; node 10
# is a SNP alternative to node 4; node 11 is an inserted segment between 6 and 7.
NODE_LEN = [
    300,  # 0  start segment (all genomes share)
    150,  # 1  shared
    200,  # 2  shared
    250,  # 3  shared
    180,  # 4  reference allele of a SNP bubble (alt = node 10)
    220,  # 5  shared
    160,  # 6  shared
    240,  # 7  shared (also the deletion bubble's "after" node)
    190,  # 8  shared
    300,  # 9  end segment (all genomes share)
    180,  # 10 ALT allele of the SNP bubble (replaces node 4)
    120,  # 11 INSERTED segment (between 6 and 7 in one haplotype)
]

# Four genome walks through the graph. Each is a list of node ids in order.
PATHS = [
    # ref     : the straight backbone, every shared node in order.
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    # hap_snp : substitutes the SNP alt (10) for the ref allele (4).
    [0, 1, 2, 3, 10, 5, 6, 7, 8, 9],
    # hap_ins : inserts node 11 between 6 and 7.
    [0, 1, 2, 3, 4, 5, 6, 11, 7, 8, 9],
    # hap_del : deletes node 5 (skips straight from 4 to 6).
    [0, 1, 2, 3, 4, 6, 7, 8, 9],
]


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic pangenome sample.")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    n = len(NODE_LEN)
    p = len(PATHS)

    lines = []
    lines.append("# SYNTHETIC pangenome graph (project 3.30). NOT real genomic data.")
    lines.append("# Format: 'N P' / N node lengths(bp) / then P lines 'L id0 id1 ...'.")
    lines.append(f"{n} {p}")
    lines.append("# node lengths (bp), one per node id 0..N-1:")
    lines.append(" ".join(str(v) for v in NODE_LEN))
    names = ["ref", "hap_snp (SNP: 10 for 4)", "hap_ins (insert 11)", "hap_del (delete 5)"]
    for walk, name in zip(PATHS, names):
        lines.append(f"# path: {name}")
        lines.append(f"{len(walk)} " + " ".join(str(v) for v in walk))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  (N={n} nodes, P={p} paths; SYNTHETIC)")


if __name__ == "__main__":
    main()
