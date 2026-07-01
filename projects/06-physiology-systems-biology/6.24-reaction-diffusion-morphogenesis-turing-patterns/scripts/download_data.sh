#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  "Fetch the full dataset" (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 6.24 : Reaction-Diffusion Morphogenesis (Turing Patterns)
#
# There is NO downloadable Turing dataset -- the data is the model configuration,
# and the pattern is produced BY the simulation. So this script downloads
# nothing: it (1) ensures the synthetic sample exists, and (2) prints where the
# optional real-biology reference images/atlases live, WITHOUT bypassing any
# registration or license (CLAUDE.md §8).
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAMPLE="$PROJECT_ROOT/data/sample/turing_params.txt"

echo "[download_data] Project 6.24 -- Reaction-Diffusion Morphogenesis (Turing Patterns)"
echo

# (1) The demo needs only the tiny synthetic config. Regenerate if it is missing.
if [[ -f "$SAMPLE" ]]; then
  echo "[download_data] Synthetic sample already present: $SAMPLE"
else
  echo "[download_data] Synthetic sample missing; regenerating via make_synthetic.py ..."
  python "$(dirname "${BASH_SOURCE[0]}")/make_synthetic.py"
fi

echo
echo "[download_data] No external dataset is required to run this project."
echo "  The 'data' is the one-line model configuration in data/sample/;"
echo "  the pattern is generated deterministically by the simulation."
echo
echo "  OPTIONAL real-biology references for visual comparison (NOT auto-downloaded;"
echo "  each has its own license / registration you must honor):"
echo "    * Pigmentation images (leopard, zebrafish): public image sources."
echo "    * HCP cortical-folding atlases: https://db.humanconnectome.org  (registration + DUA required)."
echo "    * DANDI morphogenesis imaging: https://dandiarchive.org  (open archive)."
echo
echo "  To explore other parameter regimes, sweep the synthetic config, e.g.:"
echo "    python scripts/make_synthetic.py --Dh 0.20 --steps 3000"
echo "    python scripts/make_synthetic.py --nx 128 --ny 128"
