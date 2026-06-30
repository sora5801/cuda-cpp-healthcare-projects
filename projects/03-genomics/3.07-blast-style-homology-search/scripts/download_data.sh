#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  How to get REAL protein databases (Linux/macOS)
# ---------------------------------------------------------------------------
# Project 3.7 : BLAST-Style Homology Search
#
# Prints instructions and links; downloads nothing and needs no credentials.
# Use make_synthetic.py for an offline, reproducible stand-in (CLAUDE.md sec 8).
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[download_data] Project 3.7 -- BLAST-Style Homology Search"
echo
echo "This demo searches a TINY SYNTHETIC database (data/sample/proteins_sample.fasta)."
echo "Real homology search runs against large public protein databases:"
echo
echo "  * UniRef50 / UniRef90 (clustered UniProt; the AlphaFold2 MSA database):"
echo "      https://www.uniprot.org/help/uniref"
echo "      e.g. https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz"
echo "  * NCBI nr (non-redundant proteins):"
echo "      https://ftp.ncbi.nlm.nih.gov/blast/db/"
echo "  * PDB70 (representative PDB sequences):"
echo "      https://www.rcsb.org/downloads"
echo "  * Pfam (protein-family HMMs; profile search, beyond this demo):"
echo "      https://www.ebi.ac.uk/interpro/download/"
echo
echo "These are large (UniRef50 is many GB) and licensed -- we do NOT redistribute"
echo "them. Download per the site's terms, then point the program at any FASTA:"
echo "  build/x64/Release/blast-style-homology-search.exe my_query_plus_db.fasta"
echo "(The first FASTA record is treated as the query; the rest are the database.)"
echo
echo "Offline stand-in (no download, fully reproducible):"
echo "  python scripts/make_synthetic.py --decoys 50"
echo
echo "Target data dir: $PROJECT_ROOT/data"
