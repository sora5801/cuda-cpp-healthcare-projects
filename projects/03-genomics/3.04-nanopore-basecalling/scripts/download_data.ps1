# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 3.4 : Nanopore Basecalling  (REDUCED-SCOPE: CTC greedy decode)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URLs, and
# NEVER bypasses credentials/registration. This project decodes posterior
# matrices (a neural network's OUTPUT). Real ONT data is RAW SIGNAL (.pod5 /
# .fast5), not posteriors -- turning signal into posteriors requires running a
# basecaller's network, which is the out-of-scope stage. So this script does not
# auto-download anything; it explains the sources and points to make_synthetic.py
# for an offline stand-in.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 3.4 -- Nanopore Basecalling (CTC greedy decode)"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This teaching project consumes POSTERIOR MATRICES (a network's output),"
Write-Host "not raw signal. The committed synthetic sample in data/sample/ is enough"
Write-Host "to run the demo offline; no download is required."
Write-Host ""
Write-Host "To experiment with a LARGER synthetic batch (still offline):"
Write-Host "    python scripts/make_synthetic.py --reads 4096"
Write-Host ""
Write-Host "To work with REAL nanopore data you need two things this script will not"
Write-Host "do for you (they require external tools / accounts):"
Write-Host "  1) Raw signal (.pod5 / .fast5). Public sources (respect each license):"
Write-Host "       - ONT Open Dataset (PromethION human WGS) via SRA/ENA:"
Write-Host "           https://www.ncbi.nlm.nih.gov/sra      https://www.ebi.ac.uk/ena"
Write-Host "       - R9.4.1 / R10.4.1 benchmarks (awesome-nanopore index):"
Write-Host "           https://github.com/GoekeLab/awesome-nanopore"
Write-Host "       - GIAB ONT truth sets (NA12878 / HG002):"
Write-Host "           https://www.nist.gov/programs-projects/genome-bottle"
Write-Host "       - ENA Project PRJNA594038 (multi-species ONT):"
Write-Host "           https://www.ebi.ac.uk/ena"
Write-Host "  2) A basecaller to turn that signal into posteriors (the out-of-scope"
Write-Host "     network stage): ONT Dorado -> https://github.com/nanoporetech/dorado"
Write-Host "     Dorado can emit per-step probabilities; export those in this"
Write-Host "     project's text format (see data/README.md) to feed this decoder."
Write-Host ""
Write-Host "[download_data] Nothing downloaded (by design). The demo runs on the"
Write-Host "[download_data] committed synthetic sample."
