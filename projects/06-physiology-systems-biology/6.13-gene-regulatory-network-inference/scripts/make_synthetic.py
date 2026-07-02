#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic GRN expression sample
# ---------------------------------------------------------------------------
# Project 6.13 : Gene Regulatory Network Inference (ARACNE: MI + DPI)
#
# WHY THIS EXISTS
#   Real single-cell RNA-seq (GEO, Human Cell Atlas) is large, license-bound,
#   and has no ground-truth network to check against. For a DIDACTIC demo we
#   instead build a tiny, fully SYNTHETIC dataset with a *known* regulatory
#   structure, so the learner can see the algorithm recover the right edges and
#   prune the wrong ones. Synthetic data is labeled synthetic everywhere.
#
# THE EMBEDDED GROUND TRUTH  (see ../THEORY.md and ../data/README.md)
#   Ten genes. A master transcription factor TF drives two independent cascades:
#       TF -> A -> B          (a linear chain)
#       TF -> C               (a direct target)
#       D  -> E               (a second, TF-independent regulatory pair)
#   plus F, G, H, I as unconnected NOISE genes. Because A is a deterministic-ish
#   function of TF and B of A, TF and B are correlated too -- so raw MI reports a
#   spurious TF--B edge. ARACNE's Data Processing Inequality (DPI) recognizes
#   TF--B as the weakest edge of the triangle TF-A-B and PRUNES it, recovering
#   the true chain TF->A->B. That prune is the single most interesting thing to
#   watch in the demo output.
#
#   Determinism: a fixed RNG seed makes the file byte-identical every run, so the
#   committed sample and demo/expected_output.txt never drift.
#
# USAGE
#   python scripts/make_synthetic.py                 # writes the committed sample
#   python scripts/make_synthetic.py --samples 400   # a larger synthetic set
# ===========================================================================
import argparse
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent            # the project folder
OUT = ROOT / "data" / "sample" / "expression_sample.txt"

# Gene order in the output matrix (index = row). Keep TF first so the report
# reads naturally. A,B,C,D,E carry signal; F,G,H,I are pure noise.
GENES = ["TF", "A", "B", "C", "D", "E", "F", "G", "H", "I"]


def quantize(v, levels=6):
    """Round a real value onto a small set of discrete expression LEVELS.
    Real scRNA-seq counts are already discrete-ish and noisy; quantizing here
    gives the histogram-based MI clean, well-populated bins on a small sample --
    the true dependencies then stand out sharply from the noise genes."""
    lo, hi = -3.0, 3.0
    t = (v - lo) / (hi - lo)
    t = min(max(t, 0.0), 1.0)
    return round(t * (levels - 1))


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic GRN expression sample.")
    ap.add_argument("--samples", type=int, default=200, help="number of cells (columns)")
    ap.add_argument("--seed", type=int, default=613, help="RNG seed (fixed for reproducibility)")
    ap.add_argument("--noise", type=float, default=0.35, help="stddev of transcriptional noise")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    rng = random.Random(args.seed)          # seeded -> deterministic output file
    S = args.samples
    sd = args.noise

    def noisy(x):                            # add Gaussian transcriptional noise
        return x + rng.gauss(0.0, sd)

    # Generate S independent cells; each gene's level is a noisy function of its
    # regulator(s), exactly matching the ground-truth graph in the header.
    cols = {g: [] for g in GENES}
    for _ in range(S):
        tf = rng.gauss(0.0, 1.5)             # master TF: free driver
        a  = noisy(0.9 * tf)                 # TF -> A
        b  = noisy(0.9 * a)                  # A  -> B   (so TF--B is INDIRECT)
        c  = noisy(0.9 * tf)                 # TF -> C   (direct target)
        d  = rng.gauss(0.0, 1.5)             # D: independent driver
        e  = noisy(0.9 * d)                  # D  -> E
        vals = {"TF": tf, "A": a, "B": b, "C": c, "D": d, "E": e,
                "F": rng.gauss(0, 1.5), "G": rng.gauss(0, 1.5),      # noise genes
                "H": rng.gauss(0, 1.5), "I": rng.gauss(0, 1.5)}
        for g in GENES:
            cols[g].append(quantize(vals[g]))

    # Write the loader format: "<G> <S>" then one row per gene "<name> v0 .. v(S-1)".
    lines = [f"{len(GENES)} {S}"]
    for g in GENES:
        lines.append(g + " " + " ".join(str(v) for v in cols[g]))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  (G={len(GENES)} genes, S={S} cells; SYNTHETIC)")
    print("[make_synthetic] ground truth: TF->A->B, TF->C, D->E; F,G,H,I are noise.")


if __name__ == "__main__":
    main()
