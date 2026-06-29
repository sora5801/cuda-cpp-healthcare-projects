# ===========================================================================
# scripts/download_data.ps1  --  Real MD-trajectory pointers (Windows)
# ---------------------------------------------------------------------------
# Project 1.17 : Markov State Models from MD
#
# CONTRACT (CLAUDE.md section 8): idempotent, documented, and NEVER bypasses
# credentials/registration. There is nothing to auto-download here: building an
# MSM needs a featurized MD trajectory, which you produce from raw MD with a
# tool like PyEMMA/deeptime. This script prints the pointers and the expected
# input layout; the committed synthetic sample is enough to run the demo.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 1.17 -- Markov State Models from MD"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "An MSM is built from a FEATURIZED trajectory. Featurize + tICA-reduce a"
Write-Host "raw MD run, scale a few leading components to [0,1], and write them into"
Write-Host "the format in data/README.md ('N D K lag' then N rows of D floats, in time order)."
Write-Host ""
Write-Host "  mdCATH         : https://huggingface.co/datasets/compsciencelab/mdcath  (5 us MD, 272 proteins)"
Write-Host "  Fast-folders   : chignolin / Trp-cage / Villin (Piana/Shaw, publicly shared)"
Write-Host "  GPCRmd         : https://gpcrmd.org                 (curated GPCR MD)"
Write-Host "  D. E. Shaw     : millisecond trajectories via RCSB deposition"
Write-Host "  PyEMMA         : https://github.com/markovmodel/PyEMMA   (featurize/tICA/cluster)"
Write-Host "  deeptime       : https://github.com/deeptime-ml/deeptime (modern MSM/VAMP tools)"
Write-Host ""
Write-Host "Bigger synthetic trajectory (no download):"
Write-Host "  python scripts/make_synthetic.py --frames 50000"
