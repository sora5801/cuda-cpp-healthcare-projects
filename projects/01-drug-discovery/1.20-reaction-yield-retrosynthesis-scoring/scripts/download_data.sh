#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  How to get REAL reaction data (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 1.20 : Reaction Yield / Retrosynthesis Scoring
#
# This project's "real data" is not a single file but PER-STEP FEATURES derived
# from atom-mapped reaction SMILES by a learned (transformer/GNN) yield model,
# plus candidate routes emitted by a retrosynthesis planner. This script prints
# the recipe and the source links and defers to make_synthetic.py for an offline
# stand-in (CLAUDE.md section 8). It requires no credentials and downloads
# nothing by itself.
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[download_data] Project 1.20 -- Reaction Yield / Retrosynthesis Scoring"
echo
echo "Real candidate routes + per-step features come from a planning pipeline:"
echo "  1) Get atom-mapped reactions, e.g.:"
echo "       USPTO-50k   https://github.com/connorcoley/rexgen_direct"
echo "       USPTO-MIT   https://github.com/wengong-jin/nips17-rexgen"
echo "       ORD         https://open-reaction-database.org   (open access)"
echo "     (Reaxys/CAS are commercial -- license forbids redistribution.)"
echo "  2) Run a retrosynthesis planner (AiZynthFinder / ASKCOS) on a target"
echo "     molecule to emit candidate ROUTES (sequences of reaction steps)."
echo "  3) Featurize each step with a yield model (Molecular Transformer /"
echo "     Chemformer): template_prior, precedent_count, condition_penalty,"
echo "     selectivity -- the 4 features this project scores."
echo "  4) Write the text format in data/README.md (header + shared model +"
echo "     one block per route), then score the batch with this project."
echo
echo "Offline stand-in (no download, fully reproducible):"
echo "  python scripts/make_synthetic.py --n 1000000    # a planner-scale batch"
echo
echo "Tip: keep MAX_STEPS (=6) and NUM_FEATURES (=4) consistent with"
echo "     src/route_score.h, or the loader will reject the file."
echo "Target data dir: $PROJECT_ROOT/data"
