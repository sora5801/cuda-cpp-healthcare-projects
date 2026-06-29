# THEORY — 4.11 Digital Pathology / Whole-Slide Image Analysis

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

### 4.11 Digital Pathology / Whole-Slide Image Analysis 🟡 · Active R&D
- **Deep dive:** Whole-slide images (WSIs) scanned at 40× magnification produce multi-gigapixel TIFF pyramids (0.5–5 GB per slide). Analysis requires GPU-accelerated tile extraction, feature extraction via pretrained CNNs (ResNet, ViT), and weakly supervised classification with attention-based multiple-instance learning (MIL). The tiling step alone for 10,000 slides produces ~500 million 224×224 patches; GPU DataLoaders must pipeline tile decompression, normalization, and augmentation to prevent GPU starvation. Spatial transcriptomics integration adds genomic annotations per spatial position, requiring co-registration of histology and sequencing data — a second-order GPU workload.
- **Key algorithms:** Attention-based MIL (CLAM, ABMIL), patch-level feature extraction (ResNet-50, ViT, UNI foundation model), stain normalization (Macenko, Vahadane), Otsu thresholding for tissue detection, tumor microenvironment clustering (DINO, MAE pretraining), survival prediction.
- **Datasets:** TCGA (The Cancer Genome Atlas) slides — access via GDC Data Portal (https://portal.gdc.cancer.gov/); CAMELYON16/17 lymph node metastasis detection (https://camelyon17.grand-challenge.org/); PanCancer Atlas WSIs via TCGA; TUPAC16 tumor proliferation.
- **Starter repos/tools:** CLAM (https://github.com/mahmoodlab/CLAM) — GPU-accelerated attention MIL for WSI classification, standard baseline; OpenSlide Python (https://openslide.org/) — library for reading WSI file formats; HistomicsTK (https://github.com/DigitalSlideArchive/HistomicsTK) — GPU-accelerated WSI analysis toolkit; UNI pathology foundation model (https://github.com/mahmoodlab/UNI) — pretrained ViT on 100k WSIs.
- **CUDA libraries & GPU pattern:** cuDNN (ResNet/ViT feature extraction per tile); DALI (GPU tile decode/augment pipeline); GPU-resident attention matrix for MIL (cuBLAS); batched tile inference with pinned memory transfer; multi-GPU feature extraction with `torch.multiprocessing`.

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
