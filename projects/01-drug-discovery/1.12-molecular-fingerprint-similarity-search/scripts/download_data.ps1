# ===========================================================================
# scripts/download_data.ps1  --  How to get REAL fingerprints (Windows)
# ---------------------------------------------------------------------------
# Project 1.12 : Molecular Fingerprint Similarity Search
#
# This project's "real data" is not a single file to download but fingerprints
# COMPUTED from a molecule library with RDKit. This script prints the recipe and
# defers to make_synthetic.py for an offline stand-in (CLAUDE.md section 8). It
# does not require credentials and downloads nothing by itself.
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

Write-Host "[download_data] Project 1.12 -- Molecular Fingerprint Similarity Search"
Write-Host ""
Write-Host "Real fingerprints are generated from a molecule library with RDKit:"
Write-Host "  1) Get SMILES, e.g. ChEMBL (https://www.ebi.ac.uk/chembl/) or"
Write-Host "     ZINC20 (https://zinc20.docking.org)."
Write-Host "  2) pip install rdkit"
Write-Host "  3) For each molecule: ECFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)"
Write-Host "  4) Pack each 2048-bit vector into 32 little-endian uint64 words and write the"
Write-Host "     hex format documented in data/README.md (1 query line + n library lines)."
Write-Host ""
Write-Host "Offline stand-in (no download, fully reproducible):"
Write-Host "  python scripts/make_synthetic.py --n 1000000    # a library-scale set"
Write-Host ""
Write-Host "Tip: keep FP_WORDS (=32, i.e. 2048 bits) consistent with src/reference_cpu.h."
Write-Host "Target data dir: $ProjectRoot\data"
