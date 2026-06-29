#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic route-batch sample
# ---------------------------------------------------------------------------
# Project 1.20 : Reaction Yield / Retrosynthesis Scoring
#
# WHY THIS EXISTS
#   Real reaction data (USPTO-50k, ORD, Reaxys) is either large, license-bound,
#   or needs a full transformer/GNN to turn SMILES into the per-step features we
#   score. So that the demo RUNS OFFLINE and is INTERPRETABLE, this script
#   deterministically generates a small, clearly-SYNTHETIC batch of candidate
#   retrosynthetic routes in the exact text layout the loader expects
#   (data/README.md). Synthetic data is always LABELED synthetic.
#
#   We engineer a KNOWN ANSWER (PATTERNS.md sec.6): route 0 is built to be the
#   most synthesizable (short, high template priors, low condition penalty, high
#   selectivity, fully in-stock building blocks), so the demo's headline "best
#   route is route[0]" is meaningful and verifiable -- not a coincidence.
#
#   The features and weights MUST agree with src/route_score.h:
#     features per step = [template_prior, precedent_count_norm,
#                          condition_penalty, selectivity]
#     step_yield = sigmoid(w . x + b);  route_score = PROD(step_yield) * availability
#
# USAGE
#   python scripts/make_synthetic.py                 # writes the committed sample
#   python scripts/make_synthetic.py --n 1000000     # a planner-scale batch
#   python scripts/make_synthetic.py --seed 7        # change the random routes
# ===========================================================================
import argparse
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "routes_sample.txt"

# These MUST match the compiled constants in src/route_score.h.
MAX_STEPS = 6
NUM_FEATURES = 4

# Shared logistic model (chemical intuition encoded as weights):
#   +template_prior  : reliable templates raise yield        -> positive weight
#   +precedent_count : more literature precedent -> more reliable -> positive
#   -condition_penalty: harsh conditions lower yield         -> NEGATIVE weight
#   +selectivity     : selective reactions waste less product -> positive
# The bias centers a "typical" step near a moderate yield. These numbers are
# illustrative teaching values, not fitted to real data (Limitations & honesty).
WEIGHTS = [2.5, 1.0, -3.0, 1.5]
BIAS = -0.5


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


def step_yield(x):
    """Mirror src/route_score.h::step_yield in Python (for the doc/expected check)."""
    z = BIAS + sum(w * xi for w, xi in zip(WEIGHTS, x))
    return sigmoid(z)


def route_score(steps, availability):
    """Mirror src/route_score.h::route_score (product of yields * availability)."""
    p = 1.0
    for x in steps:
        p *= step_yield(x)
    return p * availability


def make_routes(n, seed):
    """Return a list of (steps, availability) tuples.

    Route 0 is the engineered BEST route. The rest are random plausible routes
    that are individually worse (longer, harsher conditions, or scarcer leaves)
    so the ranking is stable and the demo's 'route[0] wins' is by construction.
    """
    rng = random.Random(seed)
    routes = []

    # --- Route 0: the planted winner ---------------------------------------
    # Two short, high-quality steps ending in cheap in-stock reagents.
    best_steps = [
        # template_prior, precedent_norm, condition_penalty, selectivity
        [0.95, 0.90, 0.05, 0.95],   # a textbook, well-precedented coupling
        [0.92, 0.85, 0.10, 0.90],   # a clean, selective functionalization
    ]
    routes.append((best_steps, 0.99))   # ~fully in-stock building blocks

    # --- Routes 1..n-1: random, deliberately weaker ------------------------
    for _ in range(1, n):
        num_steps = rng.randint(2, MAX_STEPS)        # longer routes -> lower product
        steps = []
        for _s in range(num_steps):
            prior   = rng.uniform(0.30, 0.85)        # generally below the winner
            prec    = rng.uniform(0.20, 0.80)
            penalty = rng.uniform(0.20, 0.80)        # harsher than the winner
            select  = rng.uniform(0.30, 0.85)
            steps.append([round(prior, 4), round(prec, 4),
                          round(penalty, 4), round(select, 4)])
        avail = round(rng.uniform(0.40, 0.95), 4)    # often scarcer leaves
        routes.append((steps, avail))
    return routes


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic route-batch sample.")
    ap.add_argument("--n", type=int, default=24, help="number of candidate routes")
    ap.add_argument("--seed", type=int, default=20, help="PRNG seed (determinism)")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    routes = make_routes(args.n, args.seed)

    lines = []
    lines.append("# SYNTHETIC route batch for project 1.20 (Reaction Yield / Retrosynthesis Scoring).")
    lines.append("# Generated by scripts/make_synthetic.py -- NOT real reaction data; do not draw")
    lines.append("# any chemical conclusion from it. Route 0 is the engineered best route.")
    lines.append("# Header: <n> <MAX_STEPS> <NUM_FEATURES>")
    lines.append(f"{args.n} {MAX_STEPS} {NUM_FEATURES}")
    lines.append("# Shared logistic model: NUM_FEATURES weights, then bias.")
    lines.append(" ".join(f"{w:g}" for w in WEIGHTS) + f" {BIAS:g}")
    for r, (steps, avail) in enumerate(routes):
        lines.append(f"# route {r}: <num_steps> <availability>, then one line per step "
                     f"({NUM_FEATURES} features)")
        lines.append(f"{len(steps)} {avail:g}")
        for x in steps:
            lines.append(" ".join(f"{v:g}" for v in x))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Echo the planted winner's score so a human can sanity-check the C++ output.
    best_steps, best_avail = routes[0]
    print(f"[make_synthetic] wrote {args.out}  (n={args.n}, seed={args.seed}; SYNTHETIC)")
    print(f"[make_synthetic] route[0] (planted best) score = "
          f"{route_score(best_steps, best_avail):.6f}")


if __name__ == "__main__":
    main()
