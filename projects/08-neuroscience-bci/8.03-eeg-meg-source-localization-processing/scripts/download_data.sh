#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Real EEG/MEG sources (Linux/macOS)
# Project 8.03 : EEG/MEG Spectral Processing (cuFFT). Downloads nothing.
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[download_data] Project 8.03 -- EEG/MEG Spectral Processing (cuFFT)"
echo
echo "Real EEG/MEG (export channels as rows of samples; prepend 'n_ch n fs'):"
echo "  PhysioNet  : https://physionet.org   (CHB-MIT, Sleep-EDF, ...)"
echo "  MNE-Python : https://mne.tools        (sample EEG/MEG + montages)"
echo "  OpenNeuro  : https://openneuro.org    (BIDS EEG/MEG datasets)"
echo
echo "Tip: use one analysis window of length n per FFT (a power of two is fastest)."
echo
echo "Longer synthetic window (no download):"
echo "  python scripts/make_synthetic.py --n 512 --fs 512"
echo
echo "Target data dir: $PROJECT_ROOT/data"
