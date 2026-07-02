# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 6.10 : Systems-Biology ODE/SDE Network Solver
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URLs, and
# NEVER bypasses credentials/registration. The real "data" for this project is
# the space of curated systems-biology MODELS (SBML files), not a single tabular
# download; this script points you to the model repositories and defers to
# scripts/make_synthetic.py for the offline teaching stand-in.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 6.10 -- Systems-Biology ODE/SDE Network Solver"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This project integrates a small GRN (the repressilator) as an ENSEMBLE."
Write-Host "The committed synthetic sample (data/sample/ensemble_params.txt) already"
Write-Host "runs the demo offline. Real curated models live in these open repositories:"
Write-Host ""
Write-Host "  * BioModels (EMBL-EBI): 1000+ curated SBML models"
Write-Host "      https://www.ebi.ac.uk/biomodels   (the repressilator is BIOMD0000000012)"
Write-Host "  * Reactome pathways:    https://reactome.org"
Write-Host "  * BioGRID network:      https://thebiogrid.org"
Write-Host "  * VCell curated models: https://vcell.org"
Write-Host ""
Write-Host "These are SBML/XML files. Turning an arbitrary SBML model into ODE RHS code"
Write-Host "is a parsing + code-generation task (see libRoadRunner / Tellurium in the"
Write-Host "README 'Prior art'); it is intentionally OUT OF SCOPE for this teaching demo,"
Write-Host "which hard-codes the repressilator RHS in src/grn.h so the focus stays on the"
Write-Host "GPU batch-ODE pattern. No credentials are required for any link above; respect"
Write-Host "each site's license before redistributing anything."
Write-Host ""
Write-Host "For a bigger SYNTHETIC sweep (more ensemble members), run:"
Write-Host "  python scripts/make_synthetic.py --na 64 --nn 64"
