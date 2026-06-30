# ===========================================================================
# scripts/download_data.ps1  --  Fetch a real density map (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 2.22 : Electron Density Map Analysis & Model Validation
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URL + format,
# and NEVER bypasses credentials/registration. The committed synthetic sample is
# enough to run the demo; this script points at the real public archives and
# fetches one open EMDB map as an example.
#
# Usage:  ./scripts/download_data.ps1 [-EmdbId 3508]
# ===========================================================================
param([string]$EmdbId = "3508")   # default: a small, openly-licensed cryo-EM entry

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"
$FullDir = Join-Path $DataDir "full"

Write-Host "[download_data] Project 2.22 -- Electron Density Map Analysis & Model Validation"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "Real data sources (all public; respect each license):"
Write-Host "  * EMDB  cryo-EM maps + half-maps : https://www.ebi.ac.uk/emdb/"
Write-Host "  * RCSB  PDB structure factors     : https://www.rcsb.org"
Write-Host "  * wwPDB OneDep validation reports : https://deposit.wwpdb.org"
Write-Host "  * IUCr  validation standards      : https://www.iucr.org  (verify URL)"
Write-Host ""
Write-Host "  The committed tiny SYNTHETIC sample in data/sample/ already runs the demo."
Write-Host "  For a larger synthetic problem:  python scripts/make_synthetic.py --n 32"
Write-Host ""

# --- Optional: fetch one open EMDB map as a real-data example -------------
# EMDB maps are gzip'd MRC/CCP4 (.map.gz). This project's loader reads a plain
# text format, so to USE a real map you would convert it first (GEMMI / mrcfile)
# -- left as an exercise in README.md. We still fetch it for inspection.
$url = "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-$EmdbId/map/emd_$EmdbId.map.gz"
$out = Join-Path $FullDir "emd_$EmdbId.map.gz"

if (Test-Path $out) {
    Write-Host "[download_data] already present (idempotent): $out"
    return
}

New-Item -ItemType Directory -Force -Path $FullDir | Out-Null
Write-Host "[download_data] fetching EMDB entry $EmdbId (open license) ..."
Write-Host "    $url"
try {
    Invoke-WebRequest -Uri $url -OutFile $out -MaximumRetryCount 3 -RetryIntervalSec 2
    Write-Host "[download_data] wrote $out"
    Write-Host "[download_data] verify size/checksum on the EMDB entry page before relying on it."
    Write-Host "[download_data] (MRC/CCP4 .map.gz -- convert to this project's text format to load it.)"
} catch {
    Write-Host "[download_data] download failed (network/entry?). The demo does not need this."
    Write-Host "[download_data] Download manually from: $url"
}
