# ===========================================================================
# scripts/download_data.ps1  --  How to get REAL synthon descriptors (Windows)
# ---------------------------------------------------------------------------
# Project 1.18 : Fragment / Combinatorial Library Enumeration
#
# This project's "real data" is not a single file to download but DESCRIPTORS
# COMPUTED from building-block SMILES with RDKit. Building-block catalogs from
# Enamine / ChemSpace require registration, so this script does NOT bypass any
# credentials (CLAUDE.md sec.8): it prints the recipe and defers to
# scripts/make_synthetic.py for an offline, reproducible stand-in. It downloads
# nothing by itself and is safe to run repeatedly (idempotent).
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

Write-Host "[download_data] Project 1.18 -- Fragment / Combinatorial Library Enumeration"
Write-Host ""
Write-Host "Real synthon descriptors are computed from building-block SMILES with RDKit:"
Write-Host "  1) Obtain a building-block catalog (registration required):"
Write-Host "       Enamine building blocks : https://enamine.net/building-blocks"
Write-Host "       Enamine REAL Space      : https://enamine.net"
Write-Host "       ChemSpace               : https://chem-space.com"
Write-Host "     Group the blocks by reactive class into 3 reactant slots (e.g. an"
Write-Host "     Ugi-like amine / aldehyde-or-acid / isocyanide-cap scheme)."
Write-Host "  2) pip install rdkit"
Write-Host "  3) For each building block, compute the 5 additive descriptors:"
Write-Host "       MW    = Descriptors.MolWt(mol)"
Write-Host "       cLogP = Crippen.MolLogP(mol)"
Write-Host "       TPSA  = rdMolDescriptors.CalcTPSA(mol)"
Write-Host "       HBD   = rdMolDescriptors.CalcNumHBD(mol)"
Write-Host "       HBA   = rdMolDescriptors.CalcNumHBA(mol)"
Write-Host "  4) Write the catalog text format documented in data/README.md"
Write-Host "     (N_SLOTS, then per slot 'SLOT k size' + one row per building block)."
Write-Host ""
Write-Host "Offline stand-in (no download, fully reproducible):"
Write-Host "  python scripts/make_synthetic.py --per-slot 40    # 40^3 = 64000 products"
Write-Host ""
Write-Host "Tip: keep N_SLOTS (=3) consistent with src/product_core.h."
Write-Host "Target data dir: $ProjectRoot\data"
