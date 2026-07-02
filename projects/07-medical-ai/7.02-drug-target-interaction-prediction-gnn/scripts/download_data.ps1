# ===========================================================================
# scripts/download_data.ps1  --  Real DTI-dataset pointers (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 7.2 : Drug-Target Interaction Prediction (GNN)
#
# There is nothing to auto-download: the demo runs fully offline on the tiny
# synthetic sample in data/sample/. This script prints where the REAL datasets
# live and how to convert them into this project's loader format (data/README.md).
# It NEVER bypasses registration/credentials (CLAUDE.md sec 8).
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

Write-Host "[download_data] Project 7.2 -- Drug-Target Interaction Prediction (GNN)"
Write-Host ""
Write-Host "The demo runs on a SYNTHETIC sample (data/sample/dti_sample.txt); nothing"
Write-Host "to fetch. Real DTI benchmarks (featurize molecular graphs + protein vectors,"
Write-Host "then write the format in data/README.md):"
Write-Host ""
Write-Host "  BindingDB : https://www.bindingdb.org/     (~2.9M measured Kd/Ki affinities)"
Write-Host "  ChEMBL    : https://www.ebi.ac.uk/chembl/   (>20M bioactivity records)"
Write-Host "  Davis     : kinase inhibitor affinities, 442 kinases x 68 drugs"
Write-Host "  KIBA      : integrated kinase-inhibitor bioactivity benchmark"
Write-Host ""
Write-Host "Toolkits that featurize + train the full model:"
Write-Host "  DeepPurpose : https://github.com/kexinhuang12345/DeepPurpose"
Write-Host "  TorchDrug   : https://github.com/DeepGraphLearning/torchdrug"
Write-Host "  DGL-LifeSci : https://github.com/awslabs/dgl-lifesci"
Write-Host ""
Write-Host "Bigger SYNTHETIC batch (no download):"
Write-Host "  python scripts/make_synthetic.py --drugs 64 --proteins 16"
Write-Host ""
Write-Host "Target data dir: $ProjectRoot\data"
