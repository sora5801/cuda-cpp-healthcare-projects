#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Point at the FULL datasets (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 1.10 -- De Novo Generative Molecular Design  (reduced-scope teaching).
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs +
# licensing, and NEVER bypasses credentials/registration. This TEACHING version
# trains on a tiny synthetic corpus and does not need the large public datasets,
# so this script only prints where to get them and defers to make_synthetic.py
# for an offline stand-in.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 1.10 -- De Novo Generative Molecular Design"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This reduced-scope teaching demo uses a SYNTHETIC corpus and needs NO"
echo "download. The committed sample (data/sample/smiles_corpus_sample.txt) is"
echo "sufficient. For a larger synthetic run:"
echo "    python scripts/make_synthetic.py --n-generate 1048576"
echo
echo "If you want to train on real public molecule corpora, fetch them yourself"
echo "and respect each license:"
echo "  * ChEMBL    (2M+ bioactive molecules)   https://www.ebi.ac.uk/chembl/      [CC-BY-SA 3.0]"
echo "  * ZINC20    (1.4B purchasable cmpds)     https://zinc20.docking.org          [free, academic use]"
echo "  * MOSES     (generation benchmark)       https://github.com/molecularsets/moses   [MIT]"
echo "  * GuacaMol  (distribution + goal bench)  https://github.com/BenevolentAI/guacamol [MIT]"
echo
echo "When wiring a real corpus, follow the idempotent pattern:"
echo "  1) skip download if the file already exists with the right checksum"
echo "  2) print source URL + expected size + SHA256"
echo "  3) for any credentialed source, print registration instructions ONLY"
