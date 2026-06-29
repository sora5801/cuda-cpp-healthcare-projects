#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate a synthetic ADMET screening dataset
# ---------------------------------------------------------------------------
# Project 1.16 : ADMET / Toxicity Prediction  (reduced-scope teaching version)
#
# WHY SYNTHETIC
#   Real ADMET labels live in Tox21 / the TDC ADMET benchmark / ClinTox (see
#   data/README.md and scripts/download_data.*), and a *trained* model is a GNN
#   that we cannot ship inside a self-contained CUDA demo. To keep the demo
#   offline, reproducible, and INTERPRETABLE we generate a clearly-SYNTHETIC
#   problem with a KNOWN answer:
#     * M=12 toxicity endpoints, each a logistic-regression model (bias + D
#       weights). Biases are tuned so flag rates spread across endpoints.
#     * N molecules, each a length-D descriptor. One molecule is planted to be
#       broadly toxic (it should top the "worst molecule" ranking); the rest are
#       drawn so their per-endpoint probabilities span the full [0,1] range.
#   A fixed RNG seed makes the output byte-for-byte reproducible, so
#   demo/expected_output.txt is stable.
#
# OUTPUT FORMAT (data/README.md):
#   line 1        : "<n> <D> <M>"
#   next M lines  : "<endpoint_name> <bias> <w_0 ... w_{D-1}>"
#   next n lines  : "<mol_name> <x_0 ... x_{D-1}>"
#   Values are written with full %.17g precision so the doubles the C++ loader
#   reads are IDENTICAL to those used here (exact text round-trip).
#
# USAGE
#   python scripts/make_synthetic.py                 # default n=24
#   python scripts/make_synthetic.py --n 1000000     # a "screening scale" set
# ===========================================================================
import argparse
import random
from pathlib import Path

D = 64    # descriptor length  -- MUST match ADMET_D in src/admet_core.h
M = 12    # toxicity endpoints -- MUST match ADMET_M in src/admet_core.h
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "admet_sample.txt"

# Human-readable endpoint labels (mirror the kind of assays in Tox21 / ADMET-AI).
# These are illustrative names only; the math does not depend on them.
ENDPOINT_NAMES = [
    "hERG_block", "Ames_mutagen", "hepatotoxicity", "Caco2_low_perm",
    "CYP3A4_inhib", "CYP2D6_inhib", "BBB_penetrant", "skin_sensitizer",
    "nephrotoxicity", "cardiotox", "clearance_high", "PAMPA_low_perm",
]

# Per-endpoint base bias: shifts the decision boundary so different endpoints
# flag different fractions of molecules (a realistic spread of prevalences).
# Negative -> harder to trip (rarer), positive -> easier (more flags).
ENDPOINT_BIAS = [
    -1.4, -0.7, -1.0, 0.2, -0.4, -1.8, 0.6, -1.1, -0.9, -0.3, 0.1, -0.6,
]


def fmt(vals):
    """Join floats at full double precision so the C++ loader reads them back
    bit-for-bit identically (no Python<->C++ rounding gap)."""
    return " ".join(f"{v:.17g}" for v in vals)


def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic ADMET dataset.")
    ap.add_argument("--n", type=int, default=24, help="number of molecules to screen")
    ap.add_argument("--seed", type=int, default=11, help="RNG seed (determinism)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    n = args.n
    rng = random.Random(args.seed)

    # --- Endpoint models: weights ~ N(0, 0.5), length D, plus the base bias. ---
    # Sparse-ish small weights keep logits in a sane range so sigmoid spans (0,1)
    # rather than saturating to 0/1 for every molecule.
    weights = [[rng.gauss(0.0, 0.5) for _ in range(D)] for _ in range(M)]
    bias = list(ENDPOINT_BIAS)

    # --- Molecule descriptors ~ N(0, 1), length D. ---------------------------
    desc = [[rng.gauss(0.0, 1.0) for _ in range(D)] for _ in range(n)]

    # Plant a broadly-toxic molecule at a shuffled position: build it to align
    # POSITIVELY with most endpoints' weight vectors (so most logits go high).
    # We take the per-component average weight direction across endpoints and
    # push the descriptor along it, scaled up so several endpoints flag it.
    toxic = [0.0] * D
    for k in range(D):
        avg_dir = sum(weights[t][k] for t in range(M)) / M
        # sign(avg_dir) * magnitude: a strong positive projection onto the mean
        # model direction -> high probability for the many endpoints aligned to it.
        toxic[k] = (1.0 if avg_dir >= 0 else -1.0) * abs(rng.gauss(2.0, 0.3))
    desc[0] = toxic                      # put it first, then shuffle so it moves

    # Molecule names BEFORE the shuffle so the toxic one keeps a stable label.
    names = [f"MOL_{i:04d}" for i in range(n)]
    order = list(range(n))
    rng.shuffle(order)
    desc = [desc[j] for j in order]
    names = [names[j] for j in order]

    # --- Write the file ------------------------------------------------------
    lines = [f"{n} {D} {M}"]
    for t in range(M):
        lines.append(f"{ENDPOINT_NAMES[t]} {bias[t]:.17g} {fmt(weights[t])}")
    for i in range(n):
        lines.append(f"{names[i]} {fmt(desc[i])}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  "
          f"(n={n}, D={D}, M={M}; SYNTHETIC, seed={args.seed})")


if __name__ == "__main__":
    main()
