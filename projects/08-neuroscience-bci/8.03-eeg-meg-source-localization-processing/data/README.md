# Data — 8.03 EEG/MEG Spectral Processing (cuFFT)

## Committed sample (`sample/eeg_sample.txt`)

| Field | Value |
|---|---|
| Origin | **Synthetic** multi-channel EEG (`scripts/make_synthetic.py`, seed 5) |
| License | Public domain (CC0) — synthetic |
| Contents | 8 channels × 256 samples, fs = 256 Hz; each channel a known dominant rhythm |

### File format

```
<n_ch> <n> <fs>            # channels, samples per channel, sampling rate (Hz)
<channel 0: n samples>     # one row per channel
<channel 1: ...>
... (n_ch rows)
```

Each channel is one or two sinusoids in a target band (delta/theta/alpha/beta/
gamma) plus low-level noise. With `fs == n`, a frequency of `f` Hz lands exactly
on FFT bin `f`, so the band powers are clean and the dominant band is obvious.

## Full dataset

Real EEG/MEG recordings come from open archives (export channels as rows):

- **PhysioNet** (<https://physionet.org>) — EEG databases (CHB-MIT, Sleep-EDF, ...).
- **MNE-Python sample data** (<https://mne.tools>) — EEG/MEG with electrode montages.
- **OpenNeuro** (<https://openneuro.org>) — BIDS-formatted EEG/MEG datasets.

`scripts/download_data.ps1` / `.sh` point to these. For a longer window:
`python scripts/make_synthetic.py --n 512 --fs 512`.

## Provenance & honesty

The sample is **synthetic** and labeled as such — clean sinusoids in known bands,
not a real recording, and of no diagnostic meaning. It exists to make the cuFFT
band-power result interpretable and verifiable.
