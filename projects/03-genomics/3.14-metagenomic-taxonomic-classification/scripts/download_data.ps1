# ===========================================================================
# scripts/download_data.ps1  --  How to get REAL reference data (Windows)
# ---------------------------------------------------------------------------
# Project 3.14 : Metagenomic Taxonomic Classification
#
# This project's "real data" is a reference k-mer database built from genomes
# plus a set of sequencing reads. There is no single file to fetch; this script
# prints the recipe and defers to make_synthetic.py for an offline stand-in
# (CLAUDE.md section 8). It requires no credentials and downloads nothing itself.
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

Write-Host "[download_data] Project 3.14 -- Metagenomic Taxonomic Classification"
Write-Host ""
Write-Host "The committed sample in data/sample/metagenome_sample.txt is SYNTHETIC and"
Write-Host "is all you need to build and run the demo offline."
Write-Host ""
Write-Host "To classify REAL reads against REAL genomes, build a reference database:"
Write-Host "  1) Reference genomes (open access):"
Write-Host "       NCBI RefSeq microbial genomes : https://ftp.ncbi.nlm.nih.gov/refseq/"
Write-Host "       (a prebuilt Kraken2 standard DB is the production form of our hash map)"
Write-Host "  2) Benchmark metagenomes with known truth (for accuracy):"
Write-Host "       CAMI challenge datasets       : https://data.cami-challenge.org/"
Write-Host "  3) Real-world reads:"
Write-Host "       Human Microbiome Project      : https://www.hmpdacc.org/"
Write-Host "       SRA metagenomics projects     : https://www.ncbi.nlm.nih.gov/sra"
Write-Host "       (HMP/SRA may require registration -- follow their instructions; this"
Write-Host "        script will NOT bypass any access control.)"
Write-Host ""
Write-Host "  4) Convert to this project's format (data/README.md):"
Write-Host "       - one REF line per taxon (taxon_name + reference sequence)"
Write-Host "       - one READ line per read (true taxon id or 0, + read sequence)"
Write-Host "     then run the demo on your file:"
Write-Host "       build\x64\Release\metagenomic-taxonomic-classification.exe your_data.txt"
Write-Host ""
Write-Host "Offline stand-in (no download, fully reproducible):"
Write-Host "  python scripts/make_synthetic.py --reads 100000    # a larger synthetic run"
Write-Host ""
Write-Host "Tip: keep KMER_K (=15) consistent with src/kmer_core.h."
Write-Host "Target data dir: $ProjectRoot\data"
