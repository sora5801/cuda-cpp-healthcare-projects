# Data — 7.10 Physiological Signal & Waveform Analysis

## Committed sample (`sample/ecg_sample.txt`)

| Field | Value |
|---|---|
| Origin | **Synthetic** ECG (`scripts/make_synthetic.py`, seed 3) |
| License | Public domain (CC0) — synthetic |
| Contents | 2048 samples, ~8 heartbeats, with noise + baseline wander |

### File format

```
<n>                 # number of samples
<sample 0>          # one float per line
<sample 1>
...
```

The signal is a sum of Gaussian P/Q/R/S/T waves per beat, plus additive
high-frequency noise and slow baseline wander — a deliberately simple stand-in
for a real ECG so the low-pass filter has something to remove.

## Full dataset

Real physiological waveforms come from open clinical archives:

- **PhysioNet** (<https://physionet.org>) — ECG/EEG/ABP/PPG databases (MIT-BIH, PTB-XL, ...).
- **MIMIC-IV Waveform** (<https://physionet.org/content/mimic4wdb/>) — ICU waveforms (credentialed).
- **MNE-Python sample data** (<https://mne.tools>) — EEG/MEG recordings.

`scripts/download_data.ps1` / `.sh` point to these (credentialed sets are **not**
bypassed). For a longer synthetic signal: `python scripts/make_synthetic.py --n 8192`.

## Provenance & honesty

The sample is **synthetic** and labeled as such; it is not a real ECG and carries
no diagnostic meaning. The point is to exercise and verify the 1-D convolution.
