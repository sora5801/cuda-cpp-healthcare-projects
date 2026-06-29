# THEORY — 8.7 EEG Seizure Detection & Prediction

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

### 8.7 EEG Seizure Detection & Prediction 🟡 · Active R&D
- **Deep dive:** Epileptic seizure prediction from scalp EEG requires continuous multi-channel spectral feature extraction and classification over rolling windows with latencies <1 s. The preictal period (minutes to hours before seizure onset) exhibits subtle changes in high-frequency oscillations (HFOs), phase-amplitude coupling, and cross-channel coherence. GPU allows real-time feature extraction from 256 channels × 2 500 Hz using cuFFT spectrograms, simultaneous CNN/LSTM classification, and sliding-window cross-correlation for connectivity graphs.
- **Key algorithms:** Short-time Fourier transform (STFT), Morlet wavelet, phase-amplitude coupling (PAC), graph-theoretic seizure propagation, 1D-CNN and BiLSTM classifiers, attention transformer for long-range EEG context, support vector machine (SVM) on spectral features, SEEG source imaging.
- **Datasets:** Temple University Hospital EEG Corpus (TUAB/TUEG) — 30 000+ EEG recordings (https://isip.piconepress.com/projects/tuh_eeg/); CHB-MIT Scalp EEG Database (PhysioNet) (https://physionet.org/content/chbmit/1.0.0/); IEEG Portal — intracranial EEG for epilepsy (https://www.ieeg.org); OpenNeuro epilepsy datasets (https://openneuro.org).
- **Starter repos/tools:** MNE-Python (https://github.com/mne-tools/mne-python) — EEG processing with parallel backend; PyTorch EEG (https://github.com/torcheeg/torcheeg) — GPU deep learning for EEG; EEGLAB (https://github.com/sccn/eeglab) — MATLAB seizure analysis plugins; BrainFlow (https://github.com/brainflow-dev/brainflow) — real-time streaming for wearable seizure monitors.
- **CUDA libraries & GPU pattern:** cuFFT batched STFT across all channels simultaneously; cuDNN for CNN classifier inference; custom CUDA kernel for phase-amplitude coupling across channel pairs; pattern: rolling window with circular GPU buffer, cuFFT on each frame, classifier inference on extracted features via TensorRT.

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
