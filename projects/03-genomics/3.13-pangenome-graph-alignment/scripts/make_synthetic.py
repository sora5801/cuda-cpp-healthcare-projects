#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate a synthetic pangenome graph + read
# ---------------------------------------------------------------------------
# Project 3.13 : Pangenome Graph Alignment
#
# Builds a tiny *variation graph* (a chain of "bubbles", each a SNP with two
# alleles) plus a query read that follows ONE chosen path through it -- so there
# is a clear, high-scoring LOCAL alignment for the graph Smith-Waterman to find,
# and the recovered NODE PATH recovers the chosen alleles. The read carries a few
# point mutations so the alignment is non-trivial (mismatches, not a perfect copy).
#
#   a0 -> [ s0ref | s0alt ] -> a1 -> [ s1ref | s1alt ] -> a2 -> ... -> aK
#              (SNP 0)                     (SNP 1)
#
# A fixed RNG seed makes the output reproducible (so expected_output.txt is
# stable). Real data is a GFA pangenome from HPRC/PGGB (see download_data.*); this
# stand-in uses the same node/edge structure in a minimal text format.
#
# OUTPUT FORMAT (consumed by src/reference_cpu.cpp::load_problem; full grammar in
# data/README.md):
#     # comment
#     Q  <DNA>                 -- the query read (exactly one)
#     N  <name> <DNA>          -- a node, listed in TOPOLOGICAL order
#     E  <src> <dst>           -- a directed edge src -> dst (must point forward)
#
# USAGE
#   python scripts/make_synthetic.py                      # default tiny graph
#   python scripts/make_synthetic.py --snps 6 --seg 8     # bigger graph
# ===========================================================================
import argparse
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "graph_sample.txt"
BASES = "ACGT"


def rand_seq(rng, k):
    """A random DNA string of length k."""
    return "".join(rng.choice(BASES) for _ in range(k))


def other_base(rng, c):
    """A random base different from c (used to make a SNP allele / a mutation)."""
    return rng.choice([b for b in BASES if b != c])


def mutate(rng, seq, rate):
    """Apply independent point substitutions to seq at the given rate."""
    return "".join(other_base(rng, c) if rng.random() < rate else c for c in seq)


def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic pangenome graph + read.")
    ap.add_argument("--snps", type=int, default=4, help="number of SNP bubbles")
    ap.add_argument("--seg", type=int, default=6, help="length of each anchor/link/allele segment")
    ap.add_argument("--mut", type=float, default=0.08, help="read mutation rate vs the chosen path")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    rng = random.Random(args.seed)

    nodes = []        # (name, dna) in topological (declaration) order
    edges = []        # (src_name, dst_name)
    chosen_path = []  # the node names the read will follow (the embedded answer)
    chosen_seq = []   # the concatenated DNA of that path

    # Start with an anchor node every haplotype shares.
    prev = "a0"
    a0 = rand_seq(rng, args.seg)
    nodes.append((prev, a0))
    chosen_path.append(prev)
    chosen_seq.append(a0)

    # Each SNP bubble: two parallel allele nodes between two anchors, then a link.
    for s in range(args.snps):
        base = rand_seq(rng, args.seg)                 # the "reference" allele
        mid = args.seg // 2
        alt = base[:mid] + other_base(rng, base[mid]) + base[mid + 1:]  # one-base variant
        ref_name = f"s{s}ref"
        alt_name = f"s{s}alt"
        nodes.append((ref_name, base))
        nodes.append((alt_name, alt))
        edges.append((prev, ref_name))
        edges.append((prev, alt_name))

        # The read follows a deterministic choice of allele (ref on even SNP,
        # alt on odd) -- this is the path we expect the aligner to recover.
        pick = ref_name if (s % 2 == 0) else alt_name
        pick_seq = base if (s % 2 == 0) else alt
        chosen_path.append(pick)
        chosen_seq.append(pick_seq)

        # A shared link/anchor node after the bubble (both alleles flow into it).
        link = f"a{s + 1}"
        link_seq = rand_seq(rng, args.seg)
        nodes.append((link, link_seq))
        edges.append((ref_name, link))
        edges.append((alt_name, link))
        chosen_path.append(link)
        chosen_seq.append(link_seq)
        prev = link

    # The read = the chosen path's DNA with a few point mutations sprinkled in.
    path_dna = "".join(chosen_seq)
    read = mutate(rng, path_dna, args.mut)

    lines = []
    lines.append("# SYNTHETIC pangenome variation graph + query read (project 3.13).")
    lines.append("# Each 's<k>ref'/'s<k>alt' pair is a SNP bubble (two alleles).")
    lines.append(f"# The read follows path: {'>'.join(chosen_path)}")
    lines.append(f"# (ref on even SNPs, alt on odd), with ~{args.mut:.0%} point mutations.")
    lines.append(f"Q {read}")
    for name, dna in nodes:
        lines.append(f"N {name} {dna}")
    for a, b in edges:
        lines.append(f"E {a} {b}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}")
    print(f"  nodes={len(nodes)} edges={len(edges)} snps={args.snps} "
          f"read_len={len(read)} mut={args.mut} seed={args.seed} (SYNTHETIC)")
    print(f"  embedded answer path: {'>'.join(chosen_path)}")


if __name__ == "__main__":
    main()
