# ===========================================================================
# scripts/download_data.ps1  --  Real EEG/MEG sources (Windows)
# ---------------------------------------------------------------------------
# Project 8.03 : EEG/MEG Spectral Processing (cuFFT). Downloads nothing.
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

Write-Host "[download_data] Project 8.03 -- EEG/MEG Spectral Processing (cuFFT)"
Write-Host ""
Write-Host "Real EEG/MEG (export channels as rows of samples; prepend 'n_ch n fs'):"
Write-Host "  PhysioNet  : https://physionet.org   (CHB-MIT, Sleep-EDF, ...)"
Write-Host "  MNE-Python : https://mne.tools        (sample EEG/MEG + montages)"
Write-Host "  OpenNeuro  : https://openneuro.org    (BIDS EEG/MEG datasets)"
Write-Host ""
Write-Host "Tip: use one analysis window of length n per FFT (a power of two is fastest)."
Write-Host ""
Write-Host "Longer synthetic window (no download):"
Write-Host "  python scripts/make_synthetic.py --n 512 --fs 512"
Write-Host ""
Write-Host "Target data dir: $ProjectRoot\data"
