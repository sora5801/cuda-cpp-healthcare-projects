# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 7.3 -- Clinical NLP over Notes & Records
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, and NEVER bypasses
# credentials/registration. The real clinical-note corpora below are ALL
# credentialed (they contain de-identified but still-protected patient text),
# so this script only prints how to obtain them legally. The committed
# SYNTHETIC sample lets the demo run offline with zero downloads.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 7.3 -- Clinical NLP over Notes & Records"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "The demo runs on the committed SYNTHETIC sample (data/sample/notes_sample.txt)."
Write-Host "The real clinical-note datasets are CREDENTIALED and cannot be auto-downloaded:"
Write-Host ""
Write-Host "  * MIMIC-IV Clinical Notes (331,794 de-identified notes, Beth Israel Deaconess)"
Write-Host "      https://physionet.org/content/mimic-iv-note/"
Write-Host "      Requires a PhysioNet credentialed account + CITI 'Data or Specimens Only"
Write-Host "      Research' training + signing the data use agreement. Do NOT bypass this."
Write-Host "  * i2b2 / n2c2 NLP Challenge datasets (NER, coreference, relation extraction)"
Write-Host "      https://n2c2.dbmi.hms.harvard.edu/   (DUA + registration required)"
Write-Host "  * MTSamples (4,999 transcribed medical reports; check license before use)"
Write-Host "      https://mtsamples.com/"
Write-Host "  * MedQA / MedMCQA (medical QA benchmarks; verify current URL/license)"
Write-Host ""
Write-Host "After you have obtained a corpus legally, tokenize it and write it into the"
Write-Host "loader's format (see scripts/make_synthetic.py + data/README.md). For a larger"
Write-Host "SYNTHETIC problem instead, run:"
Write-Host "    python scripts/make_synthetic.py --dim 16 --heads 4"
