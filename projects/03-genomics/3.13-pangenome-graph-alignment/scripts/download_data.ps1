# ===========================================================================
# scripts/download_data.ps1  --  Fetch / locate the FULL dataset (Windows)
# ---------------------------------------------------------------------------
# Project 3.13 : Pangenome Graph Alignment
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and NEVER
# bypasses credentials/registration or data-use agreements. Real pangenome graphs
# are large GFA files; this project does not redistribute them. For the demo, the
# committed SYNTHETIC sample in data/sample/ is sufficient and runs offline.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 3.13 -- Pangenome Graph Alignment"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This project ships a tiny SYNTHETIC graph (data/sample/graph_sample.txt)"
Write-Host "that is enough to build and run the demo offline. Real pangenome graphs"
Write-Host "are large and governed by data-use terms; fetch them yourself from:"
Write-Host ""
Write-Host "  HPRC (94 haplotype assemblies) : https://humanpangenome.org/"
Write-Host "  1000 Genomes (GVCFs)           : https://www.internationalgenome.org/data"
Write-Host "  Ensembl Pangenome              : https://www.ensembl.org/"
Write-Host "  PGGB tutorial graphs (small)   : https://github.com/pangenome/pggb"
Write-Host ""
Write-Host "Typical real pipeline (run on Linux with the pangenome toolkit):"
Write-Host "  1) build a graph:    pggb -i seqs.fa -o out/        # -> out/*.gfa"
Write-Host "  2) sort/inspect:     odgi sort -i out/*.gfa -o sorted.og"
Write-Host "  3) align reads:      vg giraffe -Z graph.giraffe.gbz -f reads.fq"
Write-Host ""
Write-Host "To feed a real GFA into THIS teaching program, emit one 'N <id> <seq>'"
Write-Host "line per GFA 'S' record and one 'E <src> <dst>' per 'L' record, after a"
Write-Host "topological sort (vg ids -s / odgi sort). See data/README.md."
Write-Host ""
Write-Host "For a larger SYNTHETIC problem instead, run:"
Write-Host "  python scripts/make_synthetic.py --snps 8 --seg 10"
