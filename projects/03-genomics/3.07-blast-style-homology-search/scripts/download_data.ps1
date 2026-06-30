# ===========================================================================
# scripts/download_data.ps1  --  How to get REAL protein databases (Windows)
# ---------------------------------------------------------------------------
# Project 3.7 : BLAST-Style Homology Search
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and
# NEVER bypasses credentials/registration. This demo ships a tiny SYNTHETIC
# database, so there is nothing to download to run it -- this script just prints
# where the real protein databases live and how to point the program at them.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 3.7 -- BLAST-Style Homology Search"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This demo searches a TINY SYNTHETIC database (data/sample/proteins_sample.fasta)."
Write-Host "Real homology search runs against large public protein databases:"
Write-Host ""
Write-Host "  * UniRef50 / UniRef90 (clustered UniProt; the AlphaFold2 MSA database):"
Write-Host "      https://www.uniprot.org/help/uniref"
Write-Host "      e.g. https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz"
Write-Host "  * NCBI nr (non-redundant proteins):"
Write-Host "      https://ftp.ncbi.nlm.nih.gov/blast/db/"
Write-Host "  * PDB70 (representative PDB sequences):"
Write-Host "      https://www.rcsb.org/downloads"
Write-Host "  * Pfam (protein-family HMMs; profile search, beyond this demo):"
Write-Host "      https://www.ebi.ac.uk/interpro/download/"
Write-Host ""
Write-Host "These are large (UniRef50 is many GB) and licensed -- we do NOT redistribute"
Write-Host "them. Download per the site's terms, then point the program at any FASTA"
Write-Host "(first record = query, rest = database):"
Write-Host "  build\x64\Release\blast-style-homology-search.exe my_query_plus_db.fasta"
Write-Host ""
Write-Host "Offline stand-in (no download, fully reproducible):"
Write-Host "  python scripts/make_synthetic.py --decoys 50"
