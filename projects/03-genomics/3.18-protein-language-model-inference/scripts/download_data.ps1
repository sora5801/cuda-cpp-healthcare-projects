# ===========================================================================
# scripts/download_data.ps1  --  Real protein-LM data pointers (Windows)
# ---------------------------------------------------------------------------
# Project 3.18 : Protein Language Model Inference. Nothing to download: the demo
# generates all model weights deterministically (src/attention_math.h) and ships
# a synthetic sample sequence. This script only prints where the REAL trained
# models and sequence corpora live (CLAUDE.md §8: never bypass credentials).
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

Write-Host "[download_data] Project 3.18 -- Protein Language Model Inference"
Write-Host ""
Write-Host "This teaching demo needs NO download: it generates synthetic weights in"
Write-Host "code and ships a synthetic sample sequence in data/sample/."
Write-Host ""
Write-Host "For a REAL protein language model:"
Write-Host "  Trained models (ESM-2 / ESMFold) : https://github.com/facebookresearch/esm"
Write-Host "  EvolutionaryScale ESM3           : https://github.com/evolutionaryscale/esm"
Write-Host "  Sequence corpus (UniRef50/90)    : https://www.uniprot.org/help/uniref"
Write-Host "  ESM Metagenomic Atlas            : https://esmatlas.com/"
Write-Host "  Structural validation (PDB)      : https://www.rcsb.org/"
Write-Host "  CATH / SCOP classification       : https://www.cathdb.info/"
Write-Host ""
Write-Host "ESM-2 weights are large (hundreds of MB to tens of GB) and are NOT"
Write-Host "redistributed here; fetch them via fair-esm's torch.hub / transformers APIs."
Write-Host ""
Write-Host "Longer synthetic peptide (no download):"
Write-Host "  python scripts/make_synthetic.py --len 64"
Write-Host ""
Write-Host "Target data dir: $ProjectRoot\data"
