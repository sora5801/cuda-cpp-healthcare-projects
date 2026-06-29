# ===========================================================================
# scripts/download_data.ps1  --  Real waveform sources (Windows)
# ---------------------------------------------------------------------------
# Project 7.10 : Physiological Signal & Waveform Analysis
# Prints where to obtain real ECG/EEG waveforms; downloads nothing. See sec 8.
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

Write-Host "[download_data] Project 7.10 -- Physiological Signal & Waveform Analysis"
Write-Host ""
Write-Host "Real waveforms (export one lead as a column of samples, prepend the count):"
Write-Host "  PhysioNet         : https://physionet.org           (MIT-BIH, PTB-XL ECG, ...)"
Write-Host "  MIMIC-IV Waveform : https://physionet.org/content/mimic4wdb/  (credentialed)"
Write-Host "  MNE-Python sample : https://mne.tools                (EEG/MEG)"
Write-Host ""
Write-Host "Credentialed datasets require registration; this script does NOT bypass it."
Write-Host ""
Write-Host "Longer synthetic signal (no download):"
Write-Host "  python scripts/make_synthetic.py --n 8192"
Write-Host ""
Write-Host "Target data dir: $ProjectRoot\data"
