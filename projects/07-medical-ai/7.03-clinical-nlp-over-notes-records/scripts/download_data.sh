#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 7.3 -- Clinical NLP over Notes & Records
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, and NEVER bypasses
# credentials/registration. The real clinical-note corpora below are ALL
# credentialed (de-identified but still protected patient text), so this script
# only prints how to obtain them legally. The committed SYNTHETIC sample lets
# the demo run offline with zero downloads.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 7.3 -- Clinical NLP over Notes & Records"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "The demo runs on the committed SYNTHETIC sample (data/sample/notes_sample.txt)."
echo "The real clinical-note datasets are CREDENTIALED and cannot be auto-downloaded:"
echo
echo "  * MIMIC-IV Clinical Notes (331,794 de-identified notes, Beth Israel Deaconess)"
echo "      https://physionet.org/content/mimic-iv-note/"
echo "      Requires a PhysioNet credentialed account + CITI 'Data or Specimens Only"
echo "      Research' training + signing the data use agreement. Do NOT bypass this."
echo "  * i2b2 / n2c2 NLP Challenge datasets (NER, coreference, relation extraction)"
echo "      https://n2c2.dbmi.hms.harvard.edu/   (DUA + registration required)"
echo "  * MTSamples (4,999 transcribed medical reports; check license before use)"
echo "      https://mtsamples.com/"
echo "  * MedQA / MedMCQA (medical QA benchmarks; verify current URL/license)"
echo
echo "After obtaining a corpus legally, tokenize it into the loader's format (see"
echo "scripts/make_synthetic.py + data/README.md). For a larger SYNTHETIC problem:"
echo "    python scripts/make_synthetic.py --dim 16 --heads 4"
