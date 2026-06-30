# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 3.27 -- Suffix Array / BWT / FM-Index Construction
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URL +
# expected size, and NEVER bypasses credentials/registration. The demo runs on
# the committed synthetic sample; the real genomes below are optional and only
# needed if you want to build a BWT over a real reference. This script prints
# instructions + links and defers to scripts/make_synthetic.py for an offline
# stand-in -- it does not auto-download multi-gigabyte genomes.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 3.27 -- Suffix Array / BWT / FM-Index Construction"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "The committed sample (data/sample/dna_sample.txt) is SYNTHETIC and is all"
Write-Host "the demo needs. The datasets below are the real-world targets; building a"
Write-Host "BWT over a 3 Gb genome requires the external-memory methods described in"
Write-Host "THEORY.md section 7 and is beyond this in-core teaching version."
Write-Host ""
Write-Host "Real datasets (open access -- no credentials required):"
Write-Host "  * GRCh38 human reference (~3.1 Gb):"
Write-Host "      https://www.ncbi.nlm.nih.gov/assembly/GCF_000001405.40/"
Write-Host "  * 1000 Genomes read collections (pan-read BWT):"
Write-Host "      https://www.internationalgenome.org/data"
Write-Host "  * NCBI RefSeq complete microbial genomes (small, good for scaling):"
Write-Host "      https://ftp.ncbi.nlm.nih.gov/refseq/"
Write-Host "  * Human Pangenome sequences (terabase-scale frontier):"
Write-Host "      https://humanpangenome.org/"
Write-Host ""
Write-Host "To create a larger SYNTHETIC problem that this build CAN handle in-core:"
Write-Host "    python scripts/make_synthetic.py --n 100000"
Write-Host ""
Write-Host "If you wire a real FASTA fetch here, keep it idempotent:"
Write-Host "    1) skip the download if the file already exists with the right checksum"
Write-Host "    2) strip FASTA headers/newlines to a single A/C/G/T line before use"
Write-Host "    3) print the source URL + expected size + SHA256"
