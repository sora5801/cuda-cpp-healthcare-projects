# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 1.15 : Protein-Ligand Binding Affinity Scoring (ML)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URL +
# registration steps, and NEVER bypasses credentials. PDBbind / CASF require a
# (free) account and have redistribution terms, so this script prints
# instructions only and defers to scripts/make_synthetic.py for the offline demo.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 1.15 -- Protein-Ligand Binding Affinity Scoring (ML)"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "The committed sample in data/sample/complexes_sample.txt is SYNTHETIC and is"
Write-Host "enough to build, run, and verify the demo offline. The real benchmarks below"
Write-Host "require (free) registration and are NOT auto-downloaded:"
Write-Host ""
Write-Host "  PDBbind v2020  -- 19,443 complexes with measured Kd/Ki (training set)"
Write-Host "                    http://www.pdbbind.org.cn   (register, then download)"
Write-Host "  CASF-2016      -- scoring/ranking/docking benchmark"
Write-Host "                    http://www.pdbbind.org.cn/casf.php"
Write-Host "  ChEMBL         -- bioactivity database"
Write-Host "                    https://www.ebi.ac.uk/chembl/"
Write-Host "  BindingDB      -- 2.8M measured binding affinities"
Write-Host "                    https://www.bindingdb.org"
Write-Host ""
Write-Host "To convert a real complex into this project's input format:"
Write-Host "  1) parse the protein .pdb and ligand .sdf/.mol2 (e.g. with RDKit / Biopython)"
Write-Host "  2) center a 16 A box on the binding pocket; keep atoms with element in {C,N,O,S}"
Write-Host "  3) emit one line per complex: '<m> <pKd>' then m lines '<x> <y> <z> <type> <is_ligand>'"
Write-Host "     (type: 0=C 1=N 2=O 3=S; is_ligand: 0=protein 1=ligand; coords in A in [0,16))"
Write-Host ""
Write-Host "For a larger SYNTHETIC batch to stress the GPU path instead, run:"
Write-Host "  python scripts/make_synthetic.py --n 100000"
