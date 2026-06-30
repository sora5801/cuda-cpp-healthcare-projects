#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch a real density map (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 2.22 : Electron Density Map Analysis & Model Validation
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URL + format,
# and NEVER bypasses credentials/registration. The committed synthetic sample is
# enough to run the demo; this script points at the real public archives and, if
# `curl` is available, fetches one open EMDB map as an example.
#
# Usage:  ./scripts/download_data.sh [EMDB_ID]      # e.g. ./download_data.sh 3508
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"
FULL_DIR="$DATA_DIR/full"
EMDB_ID="${1:-3508}"          # default: a small, openly-licensed cryo-EM entry

echo "[download_data] Project 2.22 -- Electron Density Map Analysis & Model Validation"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "Real data sources (all public; respect each license):"
echo "  * EMDB  cryo-EM maps + half-maps : https://www.ebi.ac.uk/emdb/"
echo "  * RCSB  PDB structure factors     : https://www.rcsb.org"
echo "  * wwPDB OneDep validation reports : https://deposit.wwpdb.org"
echo "  * IUCr  validation standards      : https://www.iucr.org  (verify URL)"
echo
echo "  The committed tiny SYNTHETIC sample in data/sample/ already runs the demo."
echo "  For a larger synthetic problem:  python scripts/make_synthetic.py --n 32"
echo

# --- Optional: fetch one open EMDB map as a real-data example -------------
# EMDB maps are gzip'd MRC/CCP4 (.map.gz). The loader in this project reads a
# plain text format, so to USE a real map you would first convert it (e.g. with
# GEMMI or mrcfile) -- left as an exercise in README.md. We still fetch it so the
# learner has a real file to inspect.
URL="https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-${EMDB_ID}/map/emd_${EMDB_ID}.map.gz"
OUT="$FULL_DIR/emd_${EMDB_ID}.map.gz"

if [[ -f "$OUT" ]]; then
  echo "[download_data] already present (idempotent): $OUT"
  exit 0
fi
if ! command -v curl >/dev/null 2>&1; then
  echo "[download_data] 'curl' not found; download manually from:"
  echo "    $URL"
  exit 0
fi

mkdir -p "$FULL_DIR"
echo "[download_data] fetching EMDB entry $EMDB_ID (open license) ..."
echo "    $URL"
curl -fL --retry 3 -o "$OUT" "$URL" || {
  echo "[download_data] download failed (network/entry?). The demo does not need this."
  exit 0
}
echo "[download_data] wrote $OUT"
echo "[download_data] verify size/checksum on the EMDB entry page before relying on it."
echo "[download_data] (MRC/CCP4 .map.gz -- convert to this project's text format to load it.)"
