# ===========================================================================
# scripts/download_data.ps1  --  How to get REAL reaction data (Windows)
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
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

Write-Host "[download_data] Project 1.20 -- Reaction Yield / Retrosynthesis Scoring"
Write-Host ""
Write-Host "Real candidate routes + per-step features come from a planning pipeline:"
Write-Host "  1) Get atom-mapped reactions, e.g.:"
Write-Host "       USPTO-50k   https://github.com/connorcoley/rexgen_direct"
Write-Host "       USPTO-MIT   https://github.com/wengong-jin/nips17-rexgen"
Write-Host "       ORD         https://open-reaction-database.org   (open access)"
Write-Host "     (Reaxys/CAS are commercial -- license forbids redistribution.)"
Write-Host "  2) Run a retrosynthesis planner (AiZynthFinder / ASKCOS) on a target"
Write-Host "     molecule to emit candidate ROUTES (sequences of reaction steps)."
Write-Host "  3) Featurize each step with a yield model (Molecular Transformer /"
Write-Host "     Chemformer): template_prior, precedent_count, condition_penalty,"
Write-Host "     selectivity -- the 4 features this project scores."
Write-Host "  4) Write the text format in data/README.md (header + shared model +"
Write-Host "     one block per route), then score the batch with this project."
Write-Host ""
Write-Host "Offline stand-in (no download, fully reproducible):"
Write-Host "  python scripts/make_synthetic.py --n 1000000    # a planner-scale batch"
Write-Host ""
Write-Host "Tip: keep MAX_STEPS (=6) and NUM_FEATURES (=4) consistent with"
Write-Host "     src/route_score.h, or the loader will reject the file."
Write-Host "Target data dir: $ProjectRoot\data"
