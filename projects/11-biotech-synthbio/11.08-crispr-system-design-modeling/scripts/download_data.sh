#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 11.8 -- CRISPR System Design & Modeling   (template skeleton)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URL + expected
# size + checksum, and NEVER bypasses credentials/registration. Defers to
# scripts/make_synthetic.py for an offline stand-in when needed.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 11.8 -- CRISPR System Design & Modeling"
echo "[download_data] Target data dir: $DATA_DIR"
echo

# TODO(impl): fill in the real dataset fetch. Template only prints guidance.
echo "TODO(impl): no full dataset wired up yet for this template skeleton."
echo "  Catalog dataset notes:"
echo "    CRISPOR Guide RNA Dataset — experimentally validated on/off-target activities (https://crispor.tefor.net/); CIRCLE-seq Off-Target Dataset (Tsai et al., Nature Methods) — unbiased off-target identification; Genome-wide CRISPR off-target benchmark (https://www.nature.com/articles/s41467-023-42695-4); ClinVar — disease-relevant on-target loci for therapeutic gRNA selection (https://www.ncbi.nlm.nih.gov/clinvar/)."
echo
echo "  The committed tiny sample in data/sample/ is enough to run the demo."
echo "  For a larger SYNTHETIC problem, run:"
echo "    python scripts/make_synthetic.py --n 1048576"
echo
echo "  When wiring a real dataset, follow this idempotent pattern:"
echo "    1) skip download if the file already exists with the right checksum"
echo "    2) print source URL + expected size + SHA256"
echo "    3) for credentialed sets, print registration instructions ONLY"
