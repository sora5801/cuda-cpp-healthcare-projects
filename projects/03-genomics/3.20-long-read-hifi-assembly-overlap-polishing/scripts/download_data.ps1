# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 3.20 : Long-Read HiFi Assembly Overlap & Polishing
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URL +
# expected size, and NEVER bypasses credentials/registration. The real HiFi
# datasets are large public archives; this script points you at them and links
# the standard sketching tools, but the committed synthetic sample already runs
# the demo end-to-end, so a download is OPTIONAL.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 3.20 -- Long-Read HiFi Assembly Overlap & Polishing"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "The committed tiny sample in data/sample/reads_sample.txt is enough to"
Write-Host "run the demo offline. No download is required for the teaching demo."
Write-Host ""
Write-Host "FULL public PacBio HiFi datasets (no credentials needed, but large):"
Write-Host "  * Human Pangenome / HG002 HiFi reads (SRA):"
Write-Host "      https://www.ncbi.nlm.nih.gov/sra  (search 'HG002 PacBio HiFi')"
Write-Host "  * Vertebrate Genomes Project HiFi assemblies:"
Write-Host "      https://vertebrategenomesproject.org/"
Write-Host "  * GenomeArk HiFi datasets (AWS open data):"
Write-Host "      https://genomeark.github.io/"
Write-Host "  * CHM13 T2T HiFi reads:"
Write-Host "      https://github.com/marbl/CHM13"
Write-Host ""
Write-Host "To turn raw HiFi reads (FASTQ/BAM) into the minimiser-sketch format this"
Write-Host "project consumes, the standard tools are:"
Write-Host "  * minimap2  (k/w minimiser sketching + overlap):  https://github.com/lh3/minimap2"
Write-Host "  * hifiasm   (state-of-the-art HiFi assembler):     https://github.com/chhylp123/hifiasm"
Write-Host ""
Write-Host "For a larger SYNTHETIC overlap graph that runs WITHOUT any download:"
Write-Host "  python scripts/make_synthetic.py --n-reads 2000"
Write-Host ""
Write-Host "When wiring a real dataset, keep the fetch idempotent: skip the download"
Write-Host "if the file already exists with the right size/checksum, and print the"
Write-Host "source URL + expected size before downloading."
