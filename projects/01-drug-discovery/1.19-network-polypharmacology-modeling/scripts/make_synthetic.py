#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic knowledge-graph sample
# ---------------------------------------------------------------------------
# Project 1.19 -- Network / Polypharmacology Modeling
#
# WHAT THIS BUILDS (and why it is SYNTHETIC)
#   The real polypharmacology knowledge graphs (STRING, DrugBank, STITCH) are
#   large and some forbid redistribution, so we cannot commit them. Instead we
#   generate a TINY, CLEARLY-SYNTHETIC drug-target knowledge graph whose TransE
#   embeddings have a KNOWN, RECOVERABLE answer baked in. The demo then uses the
#   GPU to recover that answer (rank a drug's true protein targets at the top),
#   so the learner can SEE the method work without any download.
#
# THE TransE MODEL (see ../THEORY.md for the full derivation)
#   Every entity (drug or protein) and every relation gets a d-dimensional
#   vector. A fact "(head, relation, tail)" -- e.g. (drug7, TARGETS, protein42)
#   -- is judged plausible when  head + relation ~= tail. The plausibility SCORE
#   of a candidate tail is the NEGATIVE L2 distance  -|| head + relation - tail ||
#   (closer => higher score => more plausible link). Link prediction = score the
#   query drug under the TARGETS relation against EVERY protein and rank them.
#
# HOW WE EMBED A KNOWN ANSWER (PATTERNS.md sec 6)
#   We place each protein embedding at a random base point P_j. For ONE chosen
#   query drug, we pick a small set of "true target" proteins and DEFINE the
#   drug embedding D and the relation vector R so that  D + R == P_j  EXACTLY for
#   those targets (we set D=0 and R = P_j for a single target, then nudge the
#   other true targets to sit very close). Result: the true targets have ~zero
#   distance (top score) and every other protein is farther away. The demo
#   recovers exactly that ranking -- a self-checking synthetic benchmark.
#
#   Everything here is SYNTHETIC and labeled as such (CLAUDE.md sec 8). No real
#   drug, protein, or interaction is represented; do not read clinical meaning
#   into the indices.
#
# OUTPUT LAYOUT (whitespace text; parsed by src/reference_cpu.cpp)
#   line 1:  n_proteins  dim                      (header: #candidate tails, embed dim)
#   line 2:  query drug embedding   : `dim` floats (the head vector h)
#   line 3:  TARGETS relation vector: `dim` floats (the relation vector r)
#   line 4:  n_true  then n_true protein indices   (ground-truth targets, 0-based)
#   next n_proteins lines: each protein (tail) embedding, `dim` floats
#
# USAGE
#   python scripts/make_synthetic.py                     # default tiny sample
#   python scripts/make_synthetic.py --n 4096 --dim 64   # a bigger problem
# ===========================================================================
import argparse
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent              # the project folder
OUT = ROOT / "data" / "sample" / "kg_embeddings_sample.txt"


def fmt(x: float) -> str:
    # Fixed 6-decimal text so the committed file is byte-stable across machines
    # (no platform-dependent float formatting) and round-trips cleanly to float.
    return f"{x:.6f}"


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic TransE knowledge-graph sample.")
    ap.add_argument("--n", type=int, default=64, help="number of candidate protein tails")
    ap.add_argument("--dim", type=int, default=16, help="embedding dimension d")
    ap.add_argument("--n-true", type=int, default=3, help="number of ground-truth targets")
    ap.add_argument("--seed", type=int, default=1919, help="PRNG seed (determinism)")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    n, dim, n_true = args.n, args.dim, args.n_true
    rng = random.Random(args.seed)                          # seeded => reproducible

    # --- 1. Random protein (tail) embeddings -------------------------------
    # Each protein j gets a base vector drawn uniformly from a cube. These are
    # the "tails" we will score the query drug against.
    proteins = [[rng.uniform(-1.0, 1.0) for _ in range(dim)] for _ in range(n)]

    # --- 2. Choose the ground-truth targets --------------------------------
    # Pick n_true distinct protein indices spread across the index range so the
    # answer is visibly non-trivial (not just 0,1,2). Sorted for a stable file.
    true_idx = sorted(rng.sample(range(n), n_true))
    primary = true_idx[0]                                   # the exact-match target

    # --- 3. Define the query drug head h and relation r --------------------
    # We want  h + r == protein[primary]  EXACTLY, so the primary target sits at
    # distance 0 (the unbeatable top score). Choose h as a small random vector
    # and set r = protein[primary] - h. Then for the OTHER true targets we gently
    # pull their embedding toward (h + r) so they also score near-zero distance,
    # while all the random decoys stay far away.
    h = [rng.uniform(-0.2, 0.2) for _ in range(dim)]
    r = [proteins[primary][k] - h[k] for k in range(dim)]
    target_point = [h[k] + r[k] for k in range(dim)]       # == protein[primary]

    # Nudge the secondary true targets to sit very close to target_point so they
    # rank just below the exact match but clearly above all decoys.
    for rank, j in enumerate(true_idx[1:], start=1):
        eps = 0.02 * rank                                  # small, increasing offset
        for k in range(dim):
            jitter = rng.uniform(-1.0, 1.0)
            proteins[j][k] = target_point[k] + eps * jitter

    # --- 4. Round everything to the file precision -------------------------
    # We round BEFORE writing so the in-memory ground truth matches what the
    # loader will read back (no surprise from text<->float round-off).
    def rnd(v): return [float(fmt(x)) for x in v]
    h = rnd(h); r = rnd(r)
    proteins = [rnd(p) for p in proteins]

    # --- 5. Write the file in the documented layout ------------------------
    lines = []
    lines.append(f"{n} {dim}")
    lines.append(" ".join(fmt(x) for x in h))
    lines.append(" ".join(fmt(x) for x in r))
    lines.append(str(n_true) + " " + " ".join(str(j) for j in true_idx))
    for p in proteins:
        lines.append(" ".join(fmt(x) for x in p))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Report (to stdout; this is a build-time tool, not the demo) so the author
    # can sanity-check what was embedded.
    print(f"[make_synthetic] wrote {out_path}")
    print(f"[make_synthetic]   n_proteins={n}, dim={dim}, SYNTHETIC")
    print(f"[make_synthetic]   ground-truth targets (0-based): {true_idx}")
    print(f"[make_synthetic]   primary (exact-match) target  : {primary}")


if __name__ == "__main__":
    main()
