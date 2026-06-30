# ===========================================================================
# scripts/download_data.ps1  --  Real EPR/DEER data pointers (Windows).
# ---------------------------------------------------------------------------
# Project 2.35 : Electron Paramagnetic Resonance (EPR/DEER) Constrained Modeling
#
# CONTRACT (CLAUDE.md §8): this script NEVER bypasses any registration. There is
# no single "DEER dataset" to fetch -- a real run combines a protein structure, a
# spin-label rotamer library, and an experimental P(r). So this script PRINTS the
# resources and defers to scripts/make_synthetic.py for the offline stand-in that
# the demo actually uses.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 2.35 -- Electron Paramagnetic Resonance (EPR/DEER) Constrained Modeling"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "The committed sample (data/sample/deer_sample.txt) is SYNTHETIC and is all"
Write-Host "the demo needs. To assemble a REAL DEER-restrained ensemble you need three"
Write-Host "ingredients, none of which this script downloads for you:"
Write-Host ""
Write-Host "  1) A protein structure / MD ensemble (the conformations to reweight):"
Write-Host "       PDB        https://www.rcsb.org/"
Write-Host "       SASBDB     https://www.sasbdb.org/   (EPR/SAXS-constrained models)"
Write-Host "  2) A spin-label rotamer library (e.g. MTSSL) + a DEER back-calculator:"
Write-Host "       MMM        https://www.epr.ethz.ch/software/mmm.html"
Write-Host "       DEER-PREdict (verify URL; Lindorff-Larsen lab)"
Write-Host "  3) An experimental P(r) distance distribution from a DEER/PELDOR trace:"
Write-Host "       published membrane-transporter DEER datasets; EPR.cxls community sets"
Write-Host ""
Write-Host "Reweighting reference implementation:"
Write-Host "       BioEn      https://github.com/bio-phys/BioEN"
Write-Host ""
Write-Host "Export your two label sites' rotamer clouds + your P(r) into the format in"
Write-Host "data/README.md (header must match src/deer_params.h), then run the exe on it."
Write-Host ""
Write-Host "For a larger SYNTHETIC problem (no download):"
Write-Host "    python scripts/make_synthetic.py --frames 400"
