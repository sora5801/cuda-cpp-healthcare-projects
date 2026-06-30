# ===========================================================================
# scripts/download_data.ps1  --  Pointers to the FULL datasets (Windows)
# ---------------------------------------------------------------------------
# Project 2.13 : MSA Generation Acceleration
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URLs +
# sizes, and NEVER bypasses credentials/registration. The real MSA databases are
# huge (UniRef90 ~210 GB) and are not redistributed here; this script prints
# where to get them and defers to scripts/make_synthetic.py for an offline
# stand-in. The committed tiny sample already lets the demo run with zero
# downloads.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 2.13 -- MSA Generation Acceleration"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "The committed sample (data/sample/profile_db_sample.txt) is SYNTHETIC and"
Write-Host "is all the demo needs. The real MSA databases are not redistributed here:"
Write-Host ""
Write-Host "  UniRef90    ~210 GB   https://www.uniprot.org/help/uniref"
Write-Host "  UniClust30            https://uniclust.mmseqs.com"
Write-Host "  MGnify                https://www.ebi.ac.uk/metagenomics/"
Write-Host "  BFD                   https://bfd.mmseqs.com"
Write-Host ""
Write-Host "To build a real profile HMM and search, use HMMER/HHblits or MMseqs2:"
Write-Host "  https://github.com/soedinglab/MMseqs2   (GPU-capable search/clustering)"
Write-Host "  https://github.com/sokrypton/ColabFold  (GPU MSA server for AlphaFold2)"
Write-Host ""
Write-Host "For a larger SYNTHETIC problem that runs with this project as-is:"
Write-Host "    python scripts/make_synthetic.py --n 4096 --seed 7"
Write-Host ""
Write-Host "[download_data] Nothing to download; exiting cleanly."
