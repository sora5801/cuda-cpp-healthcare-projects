#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic scRNA-seq sample
# ---------------------------------------------------------------------------
# Project 3.12 : Single-Cell RNA-seq Analysis  (reduced-scope teaching version)
#
# WHY THIS EXISTS
#   Real scRNA-seq datasets (Human Cell Atlas, 10x, CellxGene, GEO) are large and
#   variously licensed, and we want the demo to RUN OFFLINE with zero downloads.
#   So this script deterministically generates a clearly-SYNTHETIC count matrix
#   that matches the loader's layout (reference_cpu.cpp::load_dataset).
#
# WHAT IT MODELS (a cartoon of real scRNA-seq structure)
#   * 3 cell TYPES, each defined by a set of "marker" genes that are highly
#     expressed in that type and lowly expressed elsewhere. Real cell types are
#     likewise distinguished by marker-gene programs (e.g. CD3 for T cells).
#   * Per cell we draw counts from a Poisson around the type's mean profile, then
#     scale by a random LIBRARY SIZE (sequencing depth) -- the exact nuisance
#     that library-size normalization is designed to remove. Two cells of the
#     same type can therefore have very different total counts, yet normalize to
#     nearby points, so KNN connects same-type cells (high "label purity").
#   * Matrices are ~zero-inflated (most genes off in most cells), like the real
#     thing (~90% zeros).
#
#   Everything is generated from a FIXED SEED so the committed sample -- and thus
#   demo/expected_output.txt -- is byte-stable. This data is SYNTHETIC and carries
#   no clinical meaning (see data/README.md).
#
# OUTPUT FORMAT (whitespace/newline; '#'=comment), read by load_dataset():
#     N  G  k  target_sum
#     <label> c0 c1 ... c(G-1)      x N rows  (label = ground-truth type id)
#
# USAGE
#   python scripts/make_synthetic.py                  # default tiny committed sample
#   python scripts/make_synthetic.py --cells 48 --genes 24 --k 5
# ===========================================================================
import argparse
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent              # the project folder
OUT = ROOT / "data" / "sample" / "scrna_sample.txt"


def build(cells, genes, k, types, target_sum, seed):
    """Return (header, rows) for a synthetic count matrix with `types` cell
    types, each marked by a contiguous block of marker genes."""
    rng = random.Random(seed)                              # deterministic generator

    # Partition genes into `types` marker blocks. A cell of type t expresses its
    # block's genes strongly (high mean) and all other genes weakly (low mean).
    block = genes // types                                 # marker genes per type
    HIGH, LOW = 30.0, 0.6                                  # mean counts on/off marker

    # Build a mean-expression profile per type.
    profiles = []
    for t in range(types):
        mean = [LOW] * genes
        for g in range(t * block, min((t + 1) * block, genes)):
            mean[g] = HIGH
        profiles.append(mean)

    def poisson(lam):
        # Knuth's algorithm: small-lambda Poisson sample (fine for our means).
        L, k_, p = pow(2.718281828459045, -lam), 0, 1.0
        while True:
            k_ += 1
            p *= rng.random()
            if p <= L:
                return k_ - 1

    rows = []
    for c in range(cells):
        t = c % types                                      # round-robin assign types
        # Random sequencing depth multiplier in [0.5, 2.0]: the nuisance variation
        # that normalization removes. Same-type cells differ in TOTAL counts but
        # share a PROFILE, so they cluster after normalization.
        depth = 0.5 + 1.5 * rng.random()
        counts = [poisson(profiles[t][g] * depth) for g in range(genes)]
        rows.append((t, counts))

    header = f"{cells} {genes} {k} {target_sum:g}"
    return header, rows


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic scRNA-seq sample.")
    ap.add_argument("--cells", type=int, default=30, help="number of cells N")
    ap.add_argument("--genes", type=int, default=18, help="number of genes G (<= 64)")
    ap.add_argument("--k", type=int, default=5, help="neighbours per cell")
    ap.add_argument("--types", type=int, default=3, help="number of cell types")
    ap.add_argument("--target-sum", type=float, default=1.0e4,
                    help="normalization target total (CP10k by default)")
    ap.add_argument("--seed", type=int, default=20260628, help="RNG seed (determinism)")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    header, rows = build(args.cells, args.genes, args.k, args.types,
                         args.target_sum, args.seed)

    lines = [
        "# SYNTHETIC scRNA-seq count matrix -- Project 3.12 (NOT real patient data)",
        "# format: header 'N G k target_sum', then N rows of '<typeLabel> count0 .. count(G-1)'",
        f"# {args.types} cell types, marker-gene blocks, random sequencing depth per cell",
        header,
    ]
    for (t, counts) in rows:
        lines.append(str(t) + " " + " ".join(str(v) for v in counts))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  "
          f"(N={args.cells}, G={args.genes}, k={args.k}, {args.types} types; SYNTHETIC)")


if __name__ == "__main__":
    main()
