# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL/real datasets (Windows)
# ---------------------------------------------------------------------------
# Project 2.29 : Ion Channel Gating & Permeation Simulation
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and NEVER
# bypasses credentials/registration. This project's DEMO needs NO download -- it
# runs on the committed synthetic sample. This script only points you at the real
# structures and electrophysiology data you would use to ground the model, and
# offers an optional public PDB structure download as a concrete example.
#
# Usage:  ./scripts/download_data.ps1            # print guidance only
#         ./scripts/download_data.ps1 -PdbId 1BL8   # also fetch a PDB structure
# ===========================================================================
param(
    [string]$PdbId = ""   # optional: a 4-char PDB code to download (e.g. 1BL8 = KcsA)
)
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"
$FullDir = Join-Path $DataDir "full"

Write-Host "[download_data] Project 2.29 -- Ion Channel Gating & Permeation Simulation"
Write-Host "[download_data] The demo needs NO download: data/sample/channel_params.txt suffices."
Write-Host ""
Write-Host "Real sources to ground this model (respect each license):"
Write-Host "  * PDB ion-channel structures   https://www.rcsb.org   (e.g. 1BL8 KcsA, free)"
Write-Host "  * MemProtMD MD trajectories     https://memprotmd.bioch.ox.ac.uk"
Write-Host "  * Channelpedia patch-clamp data https://channelpedia.epfl.ch"
Write-Host "  * GPCRdb (GPCR/ion channels)    https://gpcrdb.org"
Write-Host ""
Write-Host "For a larger SYNTHETIC problem (no network needed), run e.g.:"
Write-Host "    python scripts/make_synthetic.py --ions 65536 --steps 20000"
Write-Host ""

if ($PdbId -ne "") {
    # Concrete, idempotent example: download one PUBLIC PDB structure. The PDB is
    # freely redistributable, so no credentials are involved. We skip the fetch if
    # the file already exists (idempotent), as the contract requires.
    New-Item -ItemType Directory -Force -Path $FullDir | Out-Null
    $dest = Join-Path $FullDir ("{0}.pdb" -f $PdbId.ToUpper())
    $url  = "https://files.rcsb.org/download/{0}.pdb" -f $PdbId.ToUpper()
    if (Test-Path $dest) {
        Write-Host "[download_data] $dest already present -- skipping (idempotent)."
    } else {
        Write-Host "[download_data] Fetching $url ..."
        Invoke-WebRequest -Uri $url -OutFile $dest
        $sha = (Get-FileHash $dest -Algorithm SHA256).Hash
        Write-Host "[download_data] Saved $dest"
        Write-Host "[download_data] SHA256 = $sha   (record this for reproducibility)"
    }
    Write-Host ""
    Write-Host "NOTE: this PDB gives the pore GEOMETRY only. Turning it into the 1-D"
    Write-Host "PMF U(z) this demo uses requires umbrella-sampling MD (see THEORY.md)."
}
