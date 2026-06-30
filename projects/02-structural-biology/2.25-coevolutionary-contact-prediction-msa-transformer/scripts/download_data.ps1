# ===========================================================================
# scripts/download_data.ps1  --  Real coevolution-MSA pointers (Windows)
# ---------------------------------------------------------------------------
# Project 2.25 : Coevolutionary Contact Prediction & MSA Transformer
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and NEVER
# bypasses credentials/registration. Real coevolution needs a DEEP MSA of a real
# protein family; building or downloading one is a multi-GB, tool-heavy step, so
# this script only prints the pointers and defers to the committed synthetic
# sample (or scripts/make_synthetic.py) for an offline, runnable demo.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 2.25 -- Coevolutionary Contact Prediction & MSA Transformer"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "Nothing is auto-downloaded. The program reads an aligned-FASTA MSA"
Write-Host "(one record per sequence, all the same length). To use a REAL family:"
Write-Host ""
Write-Host "  Pfam family MSAs   : http://pfam.xfam.org           (Stockholm -> aligned FASTA)"
Write-Host "  UniRef50/UniRef90  : https://www.uniprot.org/help/uniref  (build an MSA via jackhmmer/HHblits)"
Write-Host "  EVcouplings        : https://github.com/debbiemarkslab/EVcouplings  (benchmark families + PDB contacts)"
Write-Host "  CASP14 contacts    : https://predictioncenter.org   (community contact benchmark)"
Write-Host "  ESM-MSA-1b         : https://github.com/facebookresearch/esm  (MSA Transformer, the deep-learning route)"
Write-Host ""
Write-Host "Build an MSA, save it as aligned FASTA, then run:"
Write-Host "  build\x64\Release\coevolutionary-contact-prediction-msa-transformer.exe path\to\family.fasta"
Write-Host ""
Write-Host "No download needed for the demo -- the committed synthetic sample suffices."
Write-Host "Bigger synthetic MSA (deeper, sharper signal):"
Write-Host "  python scripts/make_synthetic.py --n 4000 --seed 7"
