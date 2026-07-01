#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  How to get REAL spectral-CT data (Linux/macOS)
# ---------------------------------------------------------------------------
# Project 4.20 : Dual-Energy / Spectral CT Reconstruction
#
# Prints pointers to real dual-energy / photon-counting CT datasets; downloads
# nothing and never bypasses registration/license gates (CLAUDE.md section 8).
# Use make_synthetic.py for an offline, reproducible stand-in.
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[download_data] Project 4.20 -- Dual-Energy / Spectral CT Reconstruction"
echo
echo "Real spectral-CT data (register / accept the license on each site):"
echo "  * AAPM Spectral CT challenge data -- verify URL at https://www.aapm.org/"
echo "  * MARS photon-counting CT datasets -- https://www.marsbioimaging.com/"
echo "  * TCIA DECT collections           -- https://www.cancerimagingarchive.net/"
echo "  * XCAT phantom simulated DECT     -- license from Duke"
echo
echo "Realistic physics inputs (to replace the analytic curves in the code):"
echo "  * NIST XCOM attenuation cross-sections -- https://physics.nist.gov/PhysRefData/Xcom/"
echo "  * SpekPy tube spectra                  -- https://bitbucket.org/spekpy/"
echo
echo "Offline stand-in (no download, fully reproducible):"
echo "  python scripts/make_synthetic.py --n 100000   # many synthetic bins"
echo
echo "Note: convert real sinograms to the simple text format in data/README.md,"
echo "or extend src/reference_cpu.cpp::load_sinogram to read your format."
echo "Target data dir: $PROJECT_ROOT/data"
