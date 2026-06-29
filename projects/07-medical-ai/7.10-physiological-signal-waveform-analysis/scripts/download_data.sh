#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Real waveform sources (Linux/macOS)
# Project 7.10 : Physiological Signal & Waveform Analysis. Downloads nothing.
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[download_data] Project 7.10 -- Physiological Signal & Waveform Analysis"
echo
echo "Real waveforms (export one lead as a column of samples, prepend the count):"
echo "  PhysioNet         : https://physionet.org           (MIT-BIH, PTB-XL ECG, ...)"
echo "  MIMIC-IV Waveform : https://physionet.org/content/mimic4wdb/  (credentialed)"
echo "  MNE-Python sample : https://mne.tools                (EEG/MEG)"
echo
echo "Credentialed datasets require registration; this script does NOT bypass it."
echo
echo "Longer synthetic signal (no download):"
echo "  python scripts/make_synthetic.py --n 8192"
echo
echo "Target data dir: $PROJECT_ROOT/data"
