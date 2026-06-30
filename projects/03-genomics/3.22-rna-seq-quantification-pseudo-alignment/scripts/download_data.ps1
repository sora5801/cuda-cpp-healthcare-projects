# ===========================================================================
# scripts/download_data.ps1  --  Point at the FULL datasets (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 3.22 : RNA-seq Quantification / Pseudo-alignment
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URLs, and
# NEVER bypasses credentials/registration. Real pseudo-alignment needs (a) a
# reference transcriptome FASTA and (b) RNA-seq FASTQs, then a tool (kallisto /
# Salmon) to PRODUCE the equivalence classes this project consumes. That pipeline
# is outside the scope of a single teaching demo, so this script only prints the
# canonical sources + the exact commands to reproduce ec counts, and otherwise
# defers to scripts/make_synthetic.py for an offline, fully-reproducible stand-in.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 3.22 -- RNA-seq Quantification / Pseudo-alignment"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This project consumes EQUIVALENCE CLASSES (ec counts), which are produced"
Write-Host "by running a pseudo-aligner on real reads. The canonical inputs are:"
Write-Host ""
Write-Host "  Reference transcriptome (FASTA):"
Write-Host "    GENCODE human transcriptome   https://www.gencodegenes.org/"
Write-Host ""
Write-Host "  RNA-seq reads (FASTQ):"
Write-Host "    ENCODE RNA-seq               https://www.encodeproject.org/"
Write-Host "    GTEx v9 tissue compendium    https://gtexportal.org/   (registration)"
Write-Host "    SRA RNA-seq studies          https://www.ncbi.nlm.nih.gov/sra"
Write-Host ""
Write-Host "  To PRODUCE ecs from those (kallisto's output includes them):"
Write-Host "    kallisto index -i idx gencode.transcripts.fa.gz"
Write-Host "    kallisto quant -i idx -o out --plaintext reads_1.fastq.gz reads_2.fastq.gz"
Write-Host "    # out/ then holds run_info.json + the ec / abundance tables to reformat"
Write-Host "    # into this project's 'T M / eff lengths / ec lines / TRUTH' text layout."
Write-Host ""
Write-Host "GTEx and some SRA studies require registration/credentials -- this script"
Write-Host "does NOT attempt to bypass that. For an offline, reproducible run, use the"
Write-Host "committed synthetic sample (already in data/sample/) or regenerate it:"
Write-Host "    python scripts/make_synthetic.py --reads 1000000"
