# ===========================================================================
# scripts/download_data.ps1  --  Fetch / point at the FULL datasets (Windows)
# ---------------------------------------------------------------------------
# Project 2.16 : Delta-Delta-G Stability Prediction
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URLs, and
# NEVER bypasses credentials/registration. The real Delta-Delta-G study sets are
# either large or governed by their own licenses, so this script only prints
# instructions + links and defers to scripts/make_synthetic.py for an offline,
# runnable stand-in. The committed tiny sample already runs the demo.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 2.16 -- Delta-Delta-G Stability Prediction"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "The committed synthetic sample (data/sample/protein_sample.txt) is all"
Write-Host "the demo needs. The datasets below are OPTIONAL study material; each has"
Write-Host "its own license -- respect it. This script does NOT download or bypass any"
Write-Host "registration; it prints where to get the data."
Write-Host ""
Write-Host "  1) Protherm / ProThermDB  (>25k experimental Delta-Delta-G values)"
Write-Host "     https://www.abren.net/protherm/   (see ProThermDB for the successor)"
Write-Host ""
Write-Host "  2) Megascale stability dataset (Rocklin lab, ~2.5M measurements)"
Write-Host "     https://github.com/Rocklin-Lab/cdna-display-proteolysis-datasets"
Write-Host ""
Write-Host "  3) ProteinGym substitution/indel benchmarks"
Write-Host "     https://github.com/OATML-Markslab/ProteinGym"
Write-Host ""
Write-Host "  4) S669 curated single-mutation stability benchmark"
Write-Host "     (verify the current canonical URL in the literature)"
Write-Host ""
Write-Host "To make a LARGER synthetic protein for scaling experiments, run e.g.:"
Write-Host "    python scripts/make_synthetic.py --residues 512 --out data/sample/protein_big.txt"
Write-Host ""
Write-Host "To turn a real PDB structure into this project's input, compute a"
Write-Host "per-residue burial fraction (relative solvent accessibility via DSSP or"
Write-Host "freesasa) and emit '<AA> <buried>' lines -- see THEORY.md."
