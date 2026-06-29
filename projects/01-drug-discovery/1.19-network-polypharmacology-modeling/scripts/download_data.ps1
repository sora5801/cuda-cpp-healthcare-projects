# ===========================================================================
# scripts/download_data.ps1  --  Pointers to the FULL datasets (Windows / PS)
# ---------------------------------------------------------------------------
# Project 1.19 -- Network / Polypharmacology Modeling
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URLs, and
# NEVER bypasses credentials/registration. The real polypharmacology knowledge
# graphs (STRING, DrugBank, STITCH, DrugComb) are large and several require
# account registration or non-redistributable licenses, so this script does NOT
# auto-download them -- it prints exactly where to get each one. The committed
# SYNTHETIC sample (data/sample/) is sufficient to run the demo offline, and
# scripts/make_synthetic.py generates larger synthetic problems on demand.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 1.19 -- Network / Polypharmacology Modeling"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This demo runs on a SYNTHETIC knowledge graph (data/sample/), so no"
Write-Host "download is required. To experiment with real polypharmacology data,"
Write-Host "obtain the sources below yourself and respect each license:"
Write-Host ""
Write-Host "  STRING PPI network   https://string-db.org/cgi/download        (CC BY 4.0; protein-protein edges + confidence scores)"
Write-Host "  DrugBank             https://go.drugbank.com/releases/latest   (requires free academic registration; drugs + targets)"
Write-Host "  STITCH               http://stitch.embl.de/cgi/download.pl     (drug-protein interactions; check per-use terms)"
Write-Host "  DrugComb             https://drugcomb.fimm.fi/                  (drug-combination synergy; cite the publication)"
Write-Host ""
Write-Host "Workflow to turn a real edge list into TransE embeddings (see THEORY.md):"
Write-Host "  1) parse edges into (head, relation, tail) triples with integer entity IDs"
Write-Host "  2) train TransE/RotatE embeddings with PyTorch Geometric or DGL on a GPU"
Write-Host "  3) export the query head + relation + all tail embeddings into this"
Write-Host "     project's text layout (data/README.md) and pass it as argv[1]"
Write-Host ""
Write-Host "For a larger SYNTHETIC problem right now, run:"
Write-Host "  python scripts/make_synthetic.py --n 100000 --dim 64"
