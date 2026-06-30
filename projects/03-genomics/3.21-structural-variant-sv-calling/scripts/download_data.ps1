# ===========================================================================
# scripts/download_data.ps1  --  Pointers to real SV benchmarks (Windows)
# ---------------------------------------------------------------------------
# Project 3.21 : Structural Variant (SV) Calling
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and NEVER
# bypasses credentials/registration. This teaching project ships a SYNTHETIC
# sample (data/sample/sv_sample.txt) and needs no download to run the demo. This
# script only points at the real gold-standard benchmarks and defers to
# scripts/make_synthetic.py for a larger offline stand-in.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 3.21 -- Structural Variant (SV) Calling"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This project runs entirely on the committed SYNTHETIC sample:"
Write-Host "    data/sample/sv_sample.txt   (one planted deletion; demo needs no download)"
Write-Host ""
Write-Host "For a LARGER synthetic problem (offline, no credentials):"
Write-Host "    python scripts/make_synthetic.py --reads 200000"
Write-Host ""
Write-Host "Real, gold-standard SV benchmarks (study these; respect each license):"
Write-Host "  * GiaB HG002 SV benchmark (NIST) -- deletion/insertion/inversion truth set:"
Write-Host "      https://www.nist.gov/programs-projects/genome-bottle"
Write-Host "      (Tier-1 VCF + BED; download via the GiaB FTP/S3 mirrors linked there.)"
Write-Host "  * PacBio sv-benchmark -- HiFi long-read SV truth + tooling:"
Write-Host "      https://github.com/PacificBiosciences/sv-benchmark"
Write-Host "  * 1000 Genomes structural-variant catalog:"
Write-Host "      https://www.internationalgenome.org/data"
Write-Host "  * ENCODE long-read SV studies:"
Write-Host "      https://www.encodeproject.org/"
Write-Host ""
Write-Host "These are BAM/CRAM + VCF, not the toy text format this teaching demo parses."
Write-Host "When wiring a real dataset, follow this idempotent pattern:"
Write-Host "  1) skip the download if the file already exists with the right SHA256"
Write-Host "  2) print the source URL + expected size + checksum before fetching"
Write-Host "  3) for any credentialed set, print registration instructions ONLY -- never"
Write-Host "     attempt to bypass authentication (CLAUDE.md section 8)."
