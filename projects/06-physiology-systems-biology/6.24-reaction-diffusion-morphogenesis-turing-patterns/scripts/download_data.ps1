# ===========================================================================
# scripts/download_data.ps1  --  "Fetch the full dataset" (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 6.24 : Reaction-Diffusion Morphogenesis (Turing Patterns)
#
# There is NO downloadable Turing dataset -- the data is the model configuration,
# and the pattern is produced BY the simulation. So this script downloads
# nothing: it (1) ensures the synthetic sample exists, and (2) prints where the
# optional real-biology *reference images/atlases* live, WITHOUT bypassing any
# registration or license (CLAUDE.md §8).
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Sample = Join-Path $ProjectRoot "data\sample\turing_params.txt"

Write-Host "[download_data] Project 6.24 -- Reaction-Diffusion Morphogenesis (Turing Patterns)"
Write-Host ""

# (1) The demo needs only the tiny synthetic config. Regenerate if it is missing.
if (Test-Path $Sample) {
    Write-Host "[download_data] Synthetic sample already present: $Sample"
} else {
    Write-Host "[download_data] Synthetic sample missing; regenerating via make_synthetic.py ..."
    python (Join-Path $PSScriptRoot "make_synthetic.py")
}

Write-Host ""
Write-Host "[download_data] No external dataset is required to run this project."
Write-Host "  The 'data' is the one-line model configuration in data/sample/;"
Write-Host "  the pattern is generated deterministically by the simulation."
Write-Host ""
Write-Host "  OPTIONAL real-biology references for visual comparison (NOT auto-downloaded;"
Write-Host "  each has its own license / registration you must honor):"
Write-Host "    * Pigmentation images (leopard, zebrafish): public image sources."
Write-Host "    * HCP cortical-folding atlases: https://db.humanconnectome.org  (registration + DUA required)."
Write-Host "    * DANDI morphogenesis imaging: https://dandiarchive.org  (open archive)."
Write-Host ""
Write-Host "  To explore other parameter regimes, sweep the synthetic config, e.g.:"
Write-Host "    python scripts/make_synthetic.py --Dh 0.20 --steps 3000"
Write-Host "    python scripts/make_synthetic.py --nx 128 --ny 128"
