# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 3.2 : Short-Read Mapping / Alignment
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URLs, and
# NEVER bypasses credentials/registration. The catalog's short-read datasets are
# either huge (1000 Genomes, SRA) or benchmark sets best fetched with their own
# tooling (GiaB, ENCODE), so this script PRINTS INSTRUCTIONS rather than blindly
# downloading gigabytes. The committed synthetic sample already runs the demo
# offline; scripts/make_synthetic.py makes a larger synthetic problem on demand.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 3.2 -- Short-Read Mapping / Alignment"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This teaching project ships a tiny SYNTHETIC sample (data/sample/reads_sample.txt)"
Write-Host "that runs the demo offline. Real short-read datasets are large and/or gated:"
Write-Host ""
Write-Host "  * 1000 Genomes Project (open, very large FASTQ/CRAM):"
Write-Host "      https://www.internationalgenome.org/data"
Write-Host "  * Genome in a Bottle (GiaB) NA12878 / HG002 benchmark WGS + truth sets:"
Write-Host "      https://www.nist.gov/programs-projects/genome-bottle"
Write-Host "      ftp://ftp-trace.ncbi.nlm.nih.gov/giab/ftp/"
Write-Host "  * SRA FASTQ archives (use the SRA Toolkit 'prefetch'/'fasterq-dump'):"
Write-Host "      https://www.ncbi.nlm.nih.gov/sra"
Write-Host "  * ENCODE ChIP/RNA-seq FASTQs (curated functional genomics):"
Write-Host "      https://www.encodeproject.org/"
Write-Host ""
Write-Host "To run this program on real data, prepare a plain-text file:"
Write-Host "    line 1            = a (short) reference sequence, ACGT only"
Write-Host "    each later line   = one read, ACGT only, all reads the same length"
Write-Host "  then:  short-read-mapping-alignment.exe <that-file>"
Write-Host ""
Write-Host "For a larger SYNTHETIC stand-in (no download, fully offline):"
Write-Host "    python scripts/make_synthetic.py --ref-len 4000 --n-reads 2000"
Write-Host ""
Write-Host "[download_data] No bytes downloaded (by design). See data/README.md."
