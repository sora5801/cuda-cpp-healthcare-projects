#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch / locate the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 3.13 : Pangenome Graph Alignment
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and NEVER
# bypasses credentials/registration or data-use agreements. Real pangenome graphs
# are large GFA files; this project does not redistribute them. For the demo, the
# committed SYNTHETIC sample in data/sample/ is sufficient and runs offline.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 3.13 -- Pangenome Graph Alignment"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This project ships a tiny SYNTHETIC graph (data/sample/graph_sample.txt)"
echo "that is enough to build and run the demo offline. Real pangenome graphs"
echo "are large and governed by data-use terms; fetch them yourself from:"
echo
echo "  HPRC (94 haplotype assemblies) : https://humanpangenome.org/"
echo "  1000 Genomes (GVCFs)           : https://www.internationalgenome.org/data"
echo "  Ensembl Pangenome              : https://www.ensembl.org/"
echo "  PGGB tutorial graphs (small)   : https://github.com/pangenome/pggb"
echo
echo "Typical real pipeline (pangenome toolkit):"
echo "  1) build a graph:    pggb -i seqs.fa -o out/        # -> out/*.gfa"
echo "  2) sort/inspect:     odgi sort -i out/*.gfa -o sorted.og"
echo "  3) align reads:      vg giraffe -Z graph.giraffe.gbz -f reads.fq"
echo
echo "To feed a real GFA into THIS teaching program, emit one 'N <id> <seq>'"
echo "line per GFA 'S' record and one 'E <src> <dst>' per 'L' record, after a"
echo "topological sort (vg ids -s / odgi sort). See data/README.md."
echo
echo "For a larger SYNTHETIC problem instead, run:"
echo "  python scripts/make_synthetic.py --snps 8 --seg 10"
