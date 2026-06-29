# ===========================================================================
# scripts/download_data.ps1  --  Pointers to the REAL training datasets (Windows)
# ---------------------------------------------------------------------------
# Project 1.35 : QMMM/ML Potential Hybrid MD   (reduced-scope teaching version)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URLs, and
# NEVER bypasses credentials/registration. This TEACHING project does not train a
# network -- its NNP weights are fixed surrogates in src/nnpmm.h -- so there is no
# dataset to fetch for the demo. This script only POINTS at the real QM/DFT
# reference datasets you would use to train a genuine NNP (MACE/NequIP), and
# defers to make_synthetic.py for the offline run config.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 1.35 -- QMMM/ML Potential Hybrid MD"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This reduced-scope teaching demo needs NO download: the NNP weights are"
Write-Host "fixed synthetic surrogates (src/nnpmm.h) and the only input is the tiny"
Write-Host "committed ensemble config (data/sample/ensemble_params.txt)."
Write-Host ""
Write-Host "To train a REAL ML potential for hybrid NNP/MM MD, use these QM reference"
Write-Host "datasets (each has its own license + access terms -- respect them):"
Write-Host "  * Transition1x : ~10M DFT calculations along reaction paths"
Write-Host "      https://zenodo.org/record/5781475"
Write-Host "  * SPICE        : drug-like + biomolecular DFT energies/forces"
Write-Host "      https://github.com/openmm/spice-dataset"
Write-Host "  * ANI-1ccx     : CCSD(T)*-quality energies (reactive extensions: verify URL)"
Write-Host ""
Write-Host "For a larger SYNTHETIC ensemble to stress the GPU path, run:"
Write-Host "  python scripts/make_synthetic.py --M 65536"
Write-Host ""
Write-Host "When wiring a real dataset, follow this idempotent pattern:"
Write-Host "  1) skip download if the file already exists with the right checksum"
Write-Host "  2) print source URL + expected size + SHA256"
Write-Host "  3) for credentialed sets, print registration instructions ONLY"
