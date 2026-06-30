#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic phylogenetics sample
# ---------------------------------------------------------------------------
# Project 3.9 : Phylogenetic Likelihood / Tree Inference
#
# WHY THIS EXISTS
#   Real curated alignments (TreeBASE, SILVA) are large and licensed; we keep the
#   demo offline and self-contained by SIMULATING a small DNA alignment down a
#   KNOWN tree, then asking the program to recover that tree by maximum
#   likelihood. Because we know the generating tree, the demo's result is
#   interpretable: the true topology should win. Everything here is SYNTHETIC and
#   labeled as such (CLAUDE.md sec 8).
#
# THE SIMULATION (matches the model in src/felsenstein.h)
#   Eight taxa related by a balanced tree of four close pairs:
#       ((t0,t1),(t2,t3))  and  ((t4,t5),(t6,t7))  joined at the root.
#   We evolve a random root sequence down each branch under the Kimura
#   2-parameter (K2P) model: along a branch of length t each site mutates with
#   probability driven by t, and a mutation is a transition (kappa x more likely)
#   or a transversion. Close pairs (short branches) end up nearly identical;
#   distant pairs differ -> a strong, recoverable phylogenetic signal.
#
#   We then emit the TRUE tree plus two WRONG resolutions of the deepest split
#   (the three trees an NNI move explores) so the program has alternatives to
#   reject. Output matches the loader grammar in data/README.md.
#
#   The PRNG is seeded (default 12345) so the file is byte-reproducible -> the
#   committed sample and demo/expected_output.txt are stable.
#
# USAGE
#   python scripts/make_synthetic.py                      # default sample
#   python scripts/make_synthetic.py --n-sites 2000 --seed 7
# ===========================================================================
import argparse
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT  = ROOT / "data" / "sample" / "phylo_sample.txt"

BASES = "ACGT"                       # index 0,1,2,3 == A,C,G,T (matches the loader)
PURINE = {0, 2}                      # A,G  (even indices) -- see felsenstein.h


def mutate(base: int, t: float, kappa: float, rng: random.Random) -> int:
    """Return the base after evolving along a branch of length t under K2P.

    We use the closed-form K2P transition probabilities (same algebra as
    src/felsenstein.h) to decide whether the site stays, makes a transition, or
    makes one of the two transversions."""
    import math
    b = 1.0 / (kappa + 2.0)          # transversion rate (a + 2b = 1)
    a = kappa * b                    # transition rate
    e_tv = math.exp(-4.0 * b * t)
    e_ts = math.exp(-2.0 * (a + b) * t)
    p_same = 0.25 + 0.25 * e_tv + 0.5 * e_ts        # P(no change)
    p_ts   = 0.25 + 0.25 * e_tv - 0.5 * e_ts        # P(the one transition)
    p_tv   = 0.25 - 0.25 * e_tv                     # P(each of two transversions)
    r = rng.random()
    if r < p_same:
        return base
    r -= p_same
    if r < p_ts:
        # transition: swap within the purine/pyrimidine class
        return {0: 2, 2: 0, 1: 3, 3: 1}[base]
    r -= p_ts
    # one of two transversions: pick the other class member by a coin flip
    if base in PURINE:
        partners = [1, 3]            # to a pyrimidine
    else:
        partners = [0, 2]            # to a purine
    return partners[0] if r < p_tv else partners[1]


def evolve(seq, t, kappa, rng):
    return [mutate(b, t, kappa, rng) for b in seq]


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic phylogenetics sample.")
    ap.add_argument("--n-sites", type=int, default=600, help="alignment length")
    ap.add_argument("--kappa", type=float, default=2.0, help="K2P transition/transversion ratio")
    ap.add_argument("--seed", type=int, default=12345, help="PRNG seed (reproducible)")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    n_sites = args.n_sites
    kappa = args.kappa

    # Random root sequence.
    root = [rng.randrange(4) for _ in range(n_sites)]

    # Balanced tree: root -> two clade ancestors -> four pair ancestors -> 8 tips.
    tip_t = 0.04        # short branch within a close pair
    pair_t = 0.10       # ancestor-of-pair to clade ancestor
    clade_t = 0.35      # clade ancestor to root (the deep split)

    cladeL = evolve(root,  clade_t, kappa, rng)
    cladeR = evolve(root,  clade_t, kappa, rng)
    pair01 = evolve(cladeL, pair_t, kappa, rng)
    pair23 = evolve(cladeL, pair_t, kappa, rng)
    pair45 = evolve(cladeR, pair_t, kappa, rng)
    pair67 = evolve(cladeR, pair_t, kappa, rng)
    tips = [
        evolve(pair01, tip_t, kappa, rng),  # t0
        evolve(pair01, tip_t, kappa, rng),  # t1
        evolve(pair23, tip_t, kappa, rng),  # t2
        evolve(pair23, tip_t, kappa, rng),  # t3
        evolve(pair45, tip_t, kappa, rng),  # t4
        evolve(pair45, tip_t, kappa, rng),  # t5
        evolve(pair67, tip_t, kappa, rng),  # t6
        evolve(pair67, tip_t, kappa, rng),  # t7
    ]
    seqs = ["".join(BASES[b] for b in tip) for tip in tips]

    # Candidate trees on leaves 0..7. n_internal = n_taxa-1 = 7, post-ordered,
    # root last. Internal node indices are 8..14 (n_taxa + k). Layout per tree:
    #   8: (a,b)   9: (c,d)   10: (8,9)   <- left clade
    #  11: (e,f)  12: (g,h)   13: (11,12) <- right clade
    #  14: (10,13)                         <- root (the deep split)
    tip_bl, pr_bl, cl_bl = 0.04, 0.10, 0.35

    def tree_block(label, a, b, c, d, e, f, g, h):
        lines = [label, "7"]
        lines.append(f"{a} {b} {tip_bl} {tip_bl}")    # node 8
        lines.append(f"{c} {d} {tip_bl} {tip_bl}")    # node 9
        lines.append(f"8 9 {pr_bl} {pr_bl}")          # node 10 (left clade)
        lines.append(f"{e} {f} {tip_bl} {tip_bl}")    # node 11
        lines.append(f"{g} {h} {tip_bl} {tip_bl}")    # node 12
        lines.append(f"11 12 {pr_bl} {pr_bl}")        # node 13 (right clade)
        lines.append(f"10 13 {cl_bl} {cl_bl}")        # node 14 (root)
        return lines

    out_lines = []
    out_lines.append("# SYNTHETIC phylogenetics sample -- generated by scripts/make_synthetic.py")
    out_lines.append("# Educational only; NOT real sequence data (CLAUDE.md sec 8).")
    out_lines.append(f"# header: n_taxa n_sites n_trees kappa   (seed={args.seed})")
    out_lines.append(f"8 {n_sites} 3 {kappa}")
    for i in range(8):
        out_lines.append(f"t{i} {seqs[i]}")
    out_lines.append("# tree 0: the TRUE generating topology -- ((t0,t1),(t2,t3)),((t4,t5),(t6,t7))")
    out_lines += tree_block("((t0,t1),(t2,t3)),((t4,t5),(t6,t7))_true", 0, 1, 2, 3, 4, 5, 6, 7)
    out_lines.append("# tree 1: WRONG deep split -- swaps clades (NNI around the root)")
    out_lines += tree_block("wrong_NNI1", 0, 1, 4, 5, 2, 3, 6, 7)
    out_lines.append("# tree 2: WRONG pairing inside a clade")
    out_lines += tree_block("wrong_NNI2", 0, 2, 1, 3, 4, 6, 5, 7)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  "
          f"(8 taxa, {n_sites} sites, 3 trees, kappa={kappa}, seed={args.seed}; SYNTHETIC)")


if __name__ == "__main__":
    main()
