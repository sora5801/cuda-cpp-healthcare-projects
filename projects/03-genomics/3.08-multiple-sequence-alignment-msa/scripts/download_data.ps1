# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 3.8 : Multiple Sequence Alignment (MSA)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URL +
# expected size + checksum, and NEVER bypasses credentials/registration. The
# committed tiny SYNTHETIC sample (data/sample/) already runs the demo offline;
# this script points at real MSA benchmarks for going further.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 3.8 -- Multiple Sequence Alignment (MSA)"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "The committed sample (data/sample/sequences_sample.fasta) is SYNTHETIC"
Write-Host "and sufficient to run the demo. No download is required for that."
Write-Host ""
Write-Host "Real MSA benchmarks you can study (verify each license before use):"
Write-Host "  * BAliBASE  -- curated reference alignments"
Write-Host "      https://www.lbgi.fr/balibase/"
Write-Host "  * HomFam    -- large homologous-family benchmark (used by Clustal Omega)"
Write-Host "      (search 'HomFam benchmark'; distributed with Clustal Omega papers)"
Write-Host "  * Pfam seed alignments -- protein family seed MSAs"
Write-Host "      https://www.ebi.ac.uk/interpro/download/  (Pfam section)"
Write-Host ""
Write-Host "These provide multi-FASTA inputs the loader reads directly (DNA mode here"
Write-Host "expects A/C/G/T only -- protein sets need the substitution-matrix upgrade"
Write-Host "described in THEORY.md before they will load)."
Write-Host ""
Write-Host "For a larger SYNTHETIC family (no download), run e.g.:"
Write-Host "    python scripts/make_synthetic.py --n 32 --sub 0.12 --indel 0.08"
Write-Host ""
Write-Host "Idempotent-download pattern to follow when wiring a real set:"
Write-Host "    1) skip the fetch if the file already exists with the right SHA256"
Write-Host "    2) print source URL + expected size + SHA256 before downloading"
Write-Host "    3) for credentialed sets, print registration instructions ONLY"
