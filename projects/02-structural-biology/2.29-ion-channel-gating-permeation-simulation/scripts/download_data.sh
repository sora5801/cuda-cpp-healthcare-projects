#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL/real datasets (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 2.29 : Ion Channel Gating & Permeation Simulation
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and NEVER
# bypasses credentials/registration. This project's DEMO needs NO download -- it
# runs on the committed synthetic sample. This script points you at the real
# structures and electrophysiology data you would use to ground the model, and
# offers an optional PUBLIC PDB structure download as a concrete example.
#
# Usage:  ./scripts/download_data.sh            # print guidance only
#         ./scripts/download_data.sh 1BL8       # also fetch a PDB structure
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"
FULL_DIR="$DATA_DIR/full"
PDB_ID="${1:-}"

echo "[download_data] Project 2.29 -- Ion Channel Gating & Permeation Simulation"
echo "[download_data] The demo needs NO download: data/sample/channel_params.txt suffices."
echo
echo "Real sources to ground this model (respect each license):"
echo "  * PDB ion-channel structures   https://www.rcsb.org   (e.g. 1BL8 KcsA, free)"
echo "  * MemProtMD MD trajectories     https://memprotmd.bioch.ox.ac.uk"
echo "  * Channelpedia patch-clamp data https://channelpedia.epfl.ch"
echo "  * GPCRdb (GPCR/ion channels)    https://gpcrdb.org"
echo
echo "For a larger SYNTHETIC problem (no network needed), run e.g.:"
echo "    python scripts/make_synthetic.py --ions 65536 --steps 20000"
echo

if [[ -n "$PDB_ID" ]]; then
  # Concrete, idempotent example: download one PUBLIC PDB structure. The PDB is
  # freely redistributable, so no credentials are involved. Skip if already present.
  mkdir -p "$FULL_DIR"
  PDB_UP="$(echo "$PDB_ID" | tr '[:lower:]' '[:upper:]')"
  DEST="$FULL_DIR/${PDB_UP}.pdb"
  URL="https://files.rcsb.org/download/${PDB_UP}.pdb"
  if [[ -f "$DEST" ]]; then
    echo "[download_data] $DEST already present -- skipping (idempotent)."
  else
    echo "[download_data] Fetching $URL ..."
    curl -fsSL "$URL" -o "$DEST"
    echo "[download_data] Saved $DEST"
    if command -v sha256sum >/dev/null 2>&1; then
      echo "[download_data] SHA256 = $(sha256sum "$DEST" | cut -d' ' -f1)   (record for reproducibility)"
    fi
  fi
  echo
  echo "NOTE: this PDB gives the pore GEOMETRY only. Turning it into the 1-D PMF"
  echo "U(z) this demo uses requires umbrella-sampling MD (see THEORY.md)."
fi
