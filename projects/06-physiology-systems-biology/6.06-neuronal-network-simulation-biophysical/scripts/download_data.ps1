# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 6.6 : Neuronal Network Simulation (Biophysical)
#
# CONTRACT (CLAUDE.md section 8): idempotent, documented, prints the source URLs,
# and NEVER bypasses credentials/registration. The demo runs entirely on the
# committed synthetic sample; the real datasets below are OPTIONAL enrichment for
# learners who want to drive the model from published morphologies/recordings.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 6.6 -- Neuronal Network Simulation (Biophysical)"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This project runs on a TINY SYNTHETIC network committed under data/sample/"
Write-Host "(network.txt). No download is required to build or run the demo."
Write-Host ""
Write-Host "OPTIONAL real-world data sources (study these; none is auto-downloaded):"
Write-Host "  * NeuroMorpho.Org  -- 200,000+ 3D neuronal reconstructions (SWC morphology)."
Write-Host "      https://neuromorpho.org   (free; cite the original reconstruction authors)"
Write-Host "  * ModelDB / modeldb.science -- curated NEURON/GENESIS model files."
Write-Host "      https://modeldb.science"
Write-Host "  * Allen Brain Cell Atlas -- patch-seq morpho-electric data."
Write-Host "      https://portal.brain-map.org"
Write-Host "  * DANDI Archive -- neurophysiology datasets (NWB format)."
Write-Host "      https://dandiarchive.org"
Write-Host ""
Write-Host "Turning an SWC morphology into this model's compartment chain is left as an"
Write-Host "exercise (see README 'Exercises'): parse the SWC tree, collapse each branch into"
Write-Host "compartments, and order them for the Hines solver."
Write-Host ""
Write-Host "For a larger SYNTHETIC ring (offline), run:"
Write-Host "    python scripts/make_synthetic.py --ncell 256 --steps 8000"
