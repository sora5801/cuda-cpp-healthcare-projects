#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch real RNA structures (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 3.10 : RNA Secondary-Structure Prediction
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and NEVER
# bypasses any site's terms. The committed synthetic sample already runs the
# demo offline; this script points you at real, public RNA databases for going
# further. It only DOWNLOADS when you opt in, and skips work already done.
#
# Usage:  ./scripts/download_data.sh            # print guidance
#         ./scripts/download_data.sh --rfam RF00001   # fetch one Rfam family
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"
FULL_DIR="$DATA_DIR/full"

echo "[download_data] Project 3.10 -- RNA Secondary-Structure Prediction"
echo "[download_data] Target data dir: $FULL_DIR"
echo

if [[ "${1:-}" == "--rfam" && -n "${2:-}" ]]; then
  ACC="$2"
  mkdir -p "$FULL_DIR"
  OUT="$FULL_DIR/${ACC}.stockholm.txt"
  if [[ -s "$OUT" ]]; then
    echo "[download_data] $OUT already exists -- skipping (idempotent)."
  else
    URL="https://rfam.org/family/${ACC}/alignment?acc=${ACC}&format=stockholm&download=1"
    echo "[download_data] Fetching Rfam family $ACC from:"
    echo "    $URL"
    # Rfam content is CC0; the Stockholm file carries a #=GC SS_cons consensus
    # secondary-structure line you can compare your predictions against.
    if command -v curl >/dev/null 2>&1; then
      curl -fsSL "$URL" -o "$OUT"
    elif command -v wget >/dev/null 2>&1; then
      wget -q "$URL" -O "$OUT"
    else
      echo "[download_data] need curl or wget on PATH." >&2; exit 1
    fi
    echo "[download_data] wrote $OUT"
  fi
  echo "[download_data] (no fixed checksum: Rfam alignments are revised over time)"
  exit 0
fi

cat <<'EOF'
No real dataset is required: the committed data/sample/rna_sample.fasta runs the
demo offline. To explore REAL RNA structures (all public, no credentials):

  * Rfam (CC0) -- RNA families + consensus structures:  https://rfam.org/
      Fetch one family's alignment here, e.g.:
        ./scripts/download_data.sh --rfam RF00001     # 5S rRNA
      The Stockholm file's "#=GC SS_cons" line is the reference structure.

  * RNAcentral (CC0) -- non-coding RNA sequences:        https://rnacentral.org/
  * PDB (CC0) -- RNA 3D structures -> secondary struct:  https://www.rcsb.org/
  * ArchiveII benchmark (curated single-seq structures), shipped with
    RNAstructure: https://rna.urmc.rochester.edu/RNAstructure.html
      Verify the mirror/terms before redistributing; we do NOT commit it.

For a longer SYNTHETIC sequence to stress the wavefront, use:
  python scripts/make_synthetic.py --random 200 --seed 1
EOF
