# THEORY — 7.10 Physiological Signal & Waveform Analysis

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. Diagrams in Mermaid/ASCII
> are welcome. See [README.md](README.md) for the quick tour and build steps.
>
> _Educational only — not for clinical use._

<!-- =======================================================================
     The block below is the verbatim catalog deep-dive for this project,
     stamped in by scaffold.py as raw material. Use it to write the sections
     that follow, then DELETE it (or fold it into "The science"). Every
     TODO(theory) below must be completed before the project is "done".
     ======================================================================= -->

<details>
<summary>Catalog deep-dive (raw source material — fold into the sections below, then remove)</summary>

### 7.10 Physiological Signal & Waveform Analysis 🟡 · Active R&D

- **Deep dive:** Processes continuous high-frequency physiological waveforms — ECG (500–2000 Hz), EEG (256–2048 Hz), arterial blood pressure, photoplethysmography — for automated diagnosis, anomaly detection, and prognostication. Long waveform segments (minutes to hours) require 1D temporal convolutions or transformer attention over thousands of time steps; both operations are GPU-bound. Processing multi-lead ECG simultaneously (12 leads × 5000 samples) as a 2D image enables CNN classification with no waveform-specific code. Batch processing of thousands of 24-hour Holter monitors in parallel on GPU is the primary throughput bottleneck in clinical annotation pipelines.
- **Key algorithms:** 1D ResNet / Inception, temporal convolutional networks (TCN), WaveNet, Bidirectional LSTM, self-supervised waveform pretraining (wav2vec 2.0 for ECG), Short-Time Fourier Transform (STFT) + CNN, multi-scale attention, event detection with anchor-free detection heads.
- **Datasets:**
  - PhysioNet Computing in Cardiology Challenge 2021 — 12-lead ECG from multiple cohorts (https://physionet.org/content/challenge-2021/)
  - MIMIC-IV-ECG — 800k+ ECGs from MIMIC patients (https://physionet.org/content/mimic-iv-ecg/)
  - PTB-XL — 21,837 12-lead ECGs with cardiologist labels (https://physionet.org/content/ptb-xl/)
  - Temple University EEG Corpus (TUEG) — 20k+ hours of clinical EEG (https://isip.piconepress.com/projects/tuh_eeg/)
- **Starter repos/tools:**
  - ECG-FM (https://github.com/bowang-lab/ecg-fm) — wav2vec-based ECG foundation model, 90M params, GPU-pretrained
  - ESI (https://github.com/comp-well-org/ESI) — multimodal ECG + text contrastive pretraining foundation model
  - CLEF ECG (https://github.com/Nokia-Bell-Labs/ecg-foundation-model) — single-lead ECG foundation model pretrained on 161k MIMIC patients
  - MNE-Python (https://github.com/mne-tools/mne-python) — EEG/MEG processing; GPU via deep learning backends
- **CUDA libraries & GPU pattern:** cuFFT for Fourier-domain convolutions on waveforms, cuDNN for 1D temporal convolutions, Flash Attention for long-sequence transformers; pattern: data-parallel batch processing across thousands of waveform windows, streaming input pipeline from waveform database.

</details>

---

## 1. The science

TODO(theory): The biology / medicine / physics being modeled — enough for a
reader to understand the *problem* before any math. What real-world question
does computing this answer?

## 2. The math

TODO(theory): The governing equations / formal problem statement, with **every
symbol defined** (units, ranges). State inputs, outputs, and the objective.

## 3. The algorithm

TODO(theory): Step-by-step. Include **complexity analysis**: serial cost vs. the
parallel work/depth. Where is the arithmetic intensity? What is the data-access
pattern?

## 4. The GPU mapping

TODO(theory): How the algorithm becomes **threads / blocks / grids**.
- Thread-to-data mapping (which thread owns which element).
- Launch configuration and the reasoning (block size, grid size).
- Memory hierarchy used and **why**: global / shared / registers / constant /
  texture. Where is the bandwidth bottleneck? What is the occupancy story?
- Which CUDA library (cuBLAS / cuFFT / cuRAND / cuSOLVER / Thrust) does what,
  and what it would take to write that step by hand (no black boxes — §6.1.6).

```
TODO(theory): an ASCII or Mermaid diagram of the grid/block decomposition.
```

## 5. Numerical considerations

TODO(theory): Precision (FP32 vs FP64) and why. Stability. Race conditions and
whether atomics are used. **Determinism**: does the parallel reduction reorder
floating-point sums? If so, say so and quantify the caveat.

## 6. How we verify correctness

TODO(theory): The CPU reference (`src/reference_cpu.cpp`), the **tolerance** and
why that value, and the edge cases checked. Explain why agreement between an
independent serial implementation and the GPU implementation is convincing
evidence of correctness.

## 7. Where this sits in the real world

TODO(theory): How production tools (named in the catalog "Prior art") do this
differently — what they add (scale, accuracy, features) that this teaching
version omits. If this is a 🔴 frontier project shipped as a reduced-scope
teaching version, describe the full approach here.

---

## References

TODO(theory): Papers, docs, and the starter repos from the catalog, with one
line each on what to learn from them.
