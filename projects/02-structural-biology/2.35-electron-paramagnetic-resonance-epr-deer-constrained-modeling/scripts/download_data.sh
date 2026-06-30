#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Real EPR/DEER data pointers (Linux / macOS).
# ---------------------------------------------------------------------------
# Project 2.35 : Electron Paramagnetic Resonance (EPR/DEER) Constrained Modeling
#
# CONTRACT (CLAUDE.md §8): this script NEVER bypasses any registration. There is
# no single "DEER dataset" to fetch -- a real run combines a protein structure, a
# spin-label rotamer library, and an experimental P(r). So it PRINTS the
# resources and defers to scripts/make_synthetic.py for the offline stand-in that
# the demo actually uses.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 2.35 -- Electron Paramagnetic Resonance (EPR/DEER) Constrained Modeling"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "The committed sample (data/sample/deer_sample.txt) is SYNTHETIC and is all"
echo "the demo needs. To assemble a REAL DEER-restrained ensemble you need three"
echo "ingredients, none of which this script downloads for you:"
echo
echo "  1) A protein structure / MD ensemble (the conformations to reweight):"
echo "       PDB        https://www.rcsb.org/"
echo "       SASBDB     https://www.sasbdb.org/   (EPR/SAXS-constrained models)"
echo "  2) A spin-label rotamer library (e.g. MTSSL) + a DEER back-calculator:"
echo "       MMM        https://www.epr.ethz.ch/software/mmm.html"
echo "       DEER-PREdict (verify URL; Lindorff-Larsen lab)"
echo "  3) An experimental P(r) distance distribution from a DEER/PELDOR trace:"
echo "       published membrane-transporter DEER datasets; EPR.cxls community sets"
echo
echo "Reweighting reference implementation:"
echo "       BioEn      https://github.com/bio-phys/BioEN"
echo
echo "Export your two label sites' rotamer clouds + your P(r) into the format in"
echo "data/README.md (header must match src/deer_params.h), then run the exe on it."
echo
echo "For a larger SYNTHETIC problem (no download):"
echo "    python scripts/make_synthetic.py --frames 400"
