# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 3.28 : Profile HMM (Viterbi / Forward)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URLs, and
# NEVER bypasses credentials/registration. The committed tiny SYNTHETIC sample in
# data/sample/ already lets the demo run offline -- this script only points you at
# the real corpora and shows how to turn them into the loader's FASTA format.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 3.28 -- Profile HMM (Viterbi / Forward)"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This teaching project ships a tiny SYNTHETIC sample (data/sample/phmm_sample.fasta)"
Write-Host "that is sufficient to build, run, and verify the demo offline. The real-world"
Write-Host "corpora below are large and/or governed by their own licenses:"
Write-Host ""
Write-Host "  Pfam-A (profile HMMs for ~20k protein families)"
Write-Host "    https://www.ebi.ac.uk/interpro/download/   (Pfam-A.hmm.gz)"
Write-Host "    License: CC0. Use HMMER's hmmemit -c to extract a family CONSENSUS,"
Write-Host "    then place it as record 0 of a FASTA file (see data/README.md)."
Write-Host ""
Write-Host "  UniRef50 (clustered protein sequences to search)"
Write-Host "    https://www.uniprot.org/help/uniref       (uniref50.fasta.gz)"
Write-Host "    License: CC BY 4.0. These become the DATABASE records (>=1 per sequence)."
Write-Host ""
Write-Host "  Rfam (RNA family profiles)   https://rfam.org/"
Write-Host "  JGI metagenome proteins      https://genome.jgi.doe.gov/  (registration required)"
Write-Host ""
Write-Host "  This project's loader expects a simple FASTA-like file:"
Write-Host "     >name <newline> AMINOACIDS <newline> ...   (record 0 = consensus)."
Write-Host "  Only the 20 standard amino acids are supported, and MAX_M=64 / MAX_L=256"
Write-Host "  (see src/phmm.h). Trim longer Pfam profiles or raise the caps + rebuild."
Write-Host ""
Write-Host "  For a larger SYNTHETIC stand-in (more decoys), run:"
Write-Host "     python scripts/make_synthetic.py --decoys 64"
