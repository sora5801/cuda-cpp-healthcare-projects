# ===========================================================================
# scripts/download_data.ps1  --  Point at the FULL datasets (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 1.10 -- De Novo Generative Molecular Design  (reduced-scope teaching).
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs +
# licensing, and NEVER bypasses credentials/registration. This TEACHING version
# trains on a tiny synthetic corpus and does not need the large public datasets,
# so this script only prints where to get them and defers to make_synthetic.py
# for an offline stand-in.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 1.10 -- De Novo Generative Molecular Design"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This reduced-scope teaching demo uses a SYNTHETIC corpus and needs NO"
Write-Host "download. The committed sample (data/sample/smiles_corpus_sample.txt) is"
Write-Host "sufficient. For a larger synthetic run:"
Write-Host "    python scripts/make_synthetic.py --n-generate 1048576"
Write-Host ""
Write-Host "If you want to train on real public molecule corpora, fetch them"
Write-Host "yourself and respect each license:"
Write-Host "  * ChEMBL    (2M+ bioactive molecules)   https://www.ebi.ac.uk/chembl/      [CC-BY-SA 3.0]"
Write-Host "  * ZINC20    (1.4B purchasable cmpds)     https://zinc20.docking.org          [free, academic use]"
Write-Host "  * MOSES     (generation benchmark)       https://github.com/molecularsets/moses   [MIT]"
Write-Host "  * GuacaMol  (distribution + goal bench)  https://github.com/BenevolentAI/guacamol [MIT]"
Write-Host ""
Write-Host "When wiring a real corpus, follow the idempotent pattern:"
Write-Host "  1) skip download if the file already exists with the right checksum"
Write-Host "  2) print source URL + expected size + SHA256"
Write-Host "  3) for any credentialed source, print registration instructions ONLY"
