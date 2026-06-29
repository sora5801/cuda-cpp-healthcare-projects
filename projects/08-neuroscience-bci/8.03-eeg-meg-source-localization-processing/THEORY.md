# THEORY — 8.3 EEG/MEG Source Localization & Processing

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

### 8.3 EEG/MEG Source Localization & Processing 🟢 · Established
- **Deep dive:** EEG/MEG source localization solves the ill-posed inverse problem of estimating the distribution of neural current sources inside the brain from measurements at 64–306 scalp/sensor locations. Forward model computation (leadfield matrix) via BEM/FEM over a realistic head model is a one-time GPU-amenable precomputation. Inverse methods range from beamforming (spatial filtering) to sparse Bayesian learning (Champagne, SESAME) with large-scale matrix factorizations that benefit from GPU. Real-time EEG filtering for BCI or epilepsy monitoring requires FIR/IIR at 1 000–10 000 Hz on 256 channels.
- **Key algorithms:** Boundary element method (BEM) for leadfield computation, minimum norm estimate (MNE), LORETA / eLORETA, beamforming (LCMV, DICS), sparse Bayesian learning, MUSIC dipole scan, dynamical statistical parametric mapping (dSPM), time-frequency analysis (Morlet wavelet, multitaper).
- **Datasets:** OpenNeuro EEG/MEG datasets in BIDS (https://openneuro.org); DANDI neurophysiology archive (https://dandiarchive.org); Human Connectome Project MEG (https://db.humanconnectome.org); TUAB / TUEG Temple University Hospital EEG corpus (https://isip.piconepress.com/projects/tuh_eeg/).
- **Starter repos/tools:** MNE-Python (https://github.com/mne-tools/mne-python) — comprehensive EEG/MEG analysis with GPU-accelerated backends; FieldTrip (https://github.com/fieldtrip/fieldtrip) — MATLAB MEG/EEG toolbox with parallel toolbox support; Brainstorm (https://github.com/brainstorm-users/brainstorm) — GUI EEG/MEG analysis; EEGLAB (https://github.com/sccn/eeglab) — MATLAB plugin ecosystem for EEG.
- **CUDA libraries & GPU pattern:** cuBLAS DGEMM for leadfield matrix multiply and beamformer weight computation; cuSOLVER for minimum-norm pseudoinverse; cuFFT for spectral analysis (all channels simultaneously); pattern: channel × time matrix operations on GPU, batch FFT across all channel pairs for coherence analysis.

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
