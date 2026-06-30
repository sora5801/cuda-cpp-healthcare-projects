# ===========================================================================
# scripts/download_data.ps1  --  Fetch real RNA structures (Windows/PowerShell)
# ---------------------------------------------------------------------------
# Project 3.10 : RNA Secondary-Structure Prediction
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and NEVER
# bypasses any site's terms. The committed synthetic sample already runs the
# demo offline; this script points you at real, public RNA databases. It only
# downloads when you opt in (-Rfam <accession>) and skips files already present.
#
# Usage:  ./scripts/download_data.ps1                 # print guidance
#         ./scripts/download_data.ps1 -Rfam RF00001   # fetch one Rfam family
# ===========================================================================
param([string]$Rfam = "")

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$FullDir = Join-Path $ProjectRoot "data\full"

Write-Host "[download_data] Project 3.10 -- RNA Secondary-Structure Prediction"
Write-Host "[download_data] Target data dir: $FullDir"
Write-Host ""

if ($Rfam -ne "") {
    New-Item -ItemType Directory -Force -Path $FullDir | Out-Null
    $out = Join-Path $FullDir "$Rfam.stockholm.txt"
    if ((Test-Path $out) -and ((Get-Item $out).Length -gt 0)) {
        Write-Host "[download_data] $out already exists -- skipping (idempotent)."
    } else {
        $url = "https://rfam.org/family/$Rfam/alignment?acc=$Rfam&format=stockholm&download=1"
        Write-Host "[download_data] Fetching Rfam family $Rfam from:"
        Write-Host "    $url"
        # Rfam content is CC0; the Stockholm file's '#=GC SS_cons' line is the
        # reference consensus secondary structure you can score against.
        Invoke-WebRequest -Uri $url -OutFile $out
        Write-Host "[download_data] wrote $out"
    }
    Write-Host "[download_data] (no fixed checksum: Rfam alignments are revised over time)"
    return
}

Write-Host @"
No real dataset is required: the committed data/sample/rna_sample.fasta runs the
demo offline. To explore REAL RNA structures (all public, no credentials):

  * Rfam (CC0) -- RNA families + consensus structures:  https://rfam.org/
      Fetch one family's alignment, e.g.:
        ./scripts/download_data.ps1 -Rfam RF00001     # 5S rRNA
      The Stockholm file's '#=GC SS_cons' line is the reference structure.

  * RNAcentral (CC0) -- non-coding RNA sequences:        https://rnacentral.org/
  * PDB (CC0) -- RNA 3D structures -> secondary struct:  https://www.rcsb.org/
  * ArchiveII benchmark (curated single-seq structures), shipped with
    RNAstructure: https://rna.urmc.rochester.edu/RNAstructure.html
      Verify the mirror/terms before redistributing; we do NOT commit it.

For a longer SYNTHETIC sequence to stress the wavefront, use:
  python scripts/make_synthetic.py --random 200 --seed 1
"@
