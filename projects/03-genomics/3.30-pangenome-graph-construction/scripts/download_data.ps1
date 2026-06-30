# ===========================================================================
# scripts/download_data.ps1  --  Fetch / build the FULL dataset (Windows)
# ---------------------------------------------------------------------------
# Project 3.30 : Pangenome Graph Construction
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and NEVER
# bypasses credentials/registration. Real pangenome graphs are BUILT from genome
# assemblies with the PGGB pipeline (not a single download), so this script prints
# the exact, reproducible recipe and links rather than fetching opaque blobs. The
# committed synthetic sample is enough to run the demo offline.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 3.30 -- Pangenome Graph Construction"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This teaching project lays out a pangenome GRAPH; real graphs are built"
Write-Host "from genome assemblies. There is no single file to download -- you build"
Write-Host "the graph with PGGB, then convert its GFA into this project's format."
Write-Host ""
Write-Host "Assembly sources (respect each license; some require registration):"
Write-Host "  * HPRC year-1 (94 human haplotypes) : https://humanpangenome.org/"
Write-Host "  * Ensembl (non-human pangenomes)    : https://www.ensembl.org/"
Write-Host "  * Vertebrate Genomes Project        : https://vertebrategenomesproject.org/"
Write-Host "  * NCBI RefSeq (bacterial pangenomes): https://ftp.ncbi.nlm.nih.gov/refseq/"
Write-Host ""
Write-Host "Reproducible recipe (run on Linux / WSL, where PGGB is supported):"
Write-Host "  1) Put your assemblies into one FASTA:  cat *.fa > seqs.fa; samtools faidx seqs.fa"
Write-Host "  2) Build the graph (Docker):"
Write-Host "       docker run -v `$PWD:/data ghcr.io/pangenome/pggb:latest \"
Write-Host "         pggb -i /data/seqs.fa -o /data/out -n <num_haplotypes> -t 8 -p 90 -s 5000"
Write-Host "  3) The graph is out/*.gfa . Convert its S (segments) and P/W (paths) lines"
Write-Host "     into this project's 'N P / lengths / paths' format (see data/README.md)."
Write-Host ""
Write-Host "For a larger SYNTHETIC graph (no download, fully offline):"
Write-Host "  python scripts/make_synthetic.py        # writes data/sample/pangenome_sample.txt"
Write-Host ""
Write-Host "[download_data] Nothing fetched. The committed synthetic sample runs the demo."
