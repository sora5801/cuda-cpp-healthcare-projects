# THEORY — 4.27 Radiomics Feature Extraction

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

A radiologist looking at a CT of a lung tumour sees things a single number cannot
capture: is the mass uniformly dense or speckled? smooth or ragged? bright at the
core or the rim? **Radiomics** is the hypothesis that these *qualitative* imaging
impressions can be captured by *quantitative* features computed from the pixels,
and that those features carry prognostic or predictive signal — sometimes as much
as an invasive biopsy ("virtual biopsy"). A radiomics study:

1. **Segments** the region of interest (ROI) — the tumour or organ — producing a
   binary mask over the 3-D image.
2. **Extracts features** from the masked voxels: shape, first-order intensity
   statistics, and *texture* (spatial arrangement of intensities).
3. **Models** outcome (survival, response, mutation status) from the feature
   vectors across a cohort of hundreds to thousands of patients.

Step 2 is where the compute lives, and texture features are the expensive part.
The single most-used texture descriptor is the **Gray-Level Co-occurrence Matrix
(GLCM)** of Haralick (1973): it summarizes *how pairs of neighbouring voxels'
intensities relate*, which is exactly the "speckled vs. smooth" question. Because
a cohort has thousands of scans and each ROI can hold ~10⁶ voxels, radiomics
feature extraction is a genuine throughput problem — the motivation for a GPU.

## 2. The math

**Input.** A 3-D image $I(x,y,z)$ on a grid $n_x \times n_y \times n_z$, and a
binary mask $M(x,y,z)\in\{0,1\}$ selecting the ROI. Let $R=\{v : M(v)=1\}$ be the
set of ROI voxels, $|R|=n_\text{roi}$.

**Discretization.** Raw intensities span a wide continuous range, so we bin them
into $N_g$ discrete gray levels (here $N_g=8$). With ROI range
$[v_\min, v_\max]$, the fixed-bin-count map is

$$
g(v) = \min\!\Big(N_g-1,\ \big\lfloor \tfrac{I(v)-v_\min}{v_\max-v_\min}\,N_g \big\rfloor\Big)\in\{0,\dots,N_g-1\}.
$$

**First-order features** come from the ROI histogram $h[k]=|\{v\in R: g(v)=k\}|$
and its probability $p[k]=h[k]/n_\text{roi}$:

$$
\mu=\sum_k k\,p[k],\quad
\sigma^2=\sum_k (k-\mu)^2 p[k],\quad
\text{energy}=\sum_k h[k]\,k^2,\quad
H_1=-\!\sum_{k:p>0} p[k]\log_2 p[k].
$$

**The GLCM.** For a displacement (direction) $\delta$, the co-occurrence matrix
$P_\delta$ is an $N_g\times N_g$ table where

$$
P_\delta[i][j] = \big|\{(v,w): v,w\in R,\ w=v+\delta,\ g(v)=i,\ g(w)=j\}\big|.
$$

A voxel in 3-D has 26 neighbours; each direction and its opposite carry the same
information once we **symmetrize** ($P \leftarrow P + P^\top$), so only **13
distinct directions** matter. We sum over them, $P=\sum_{\delta} (P_\delta +
P_\delta^\top)$, then **normalize** to a probability matrix $\hat P = P/\sum P$.
Haralick's scalars follow (with marginal mean $\mu_g=\sum_i i\sum_j \hat P[i][j]$
and std $\sigma_g$):

$$
\text{contrast}=\sum_{i,j}\hat P[i][j](i-j)^2,\quad
\text{ASM/energy}=\sum_{i,j}\hat P[i][j]^2,\quad
\text{homogeneity}=\sum_{i,j}\frac{\hat P[i][j]}{1+(i-j)^2},
$$
$$
\text{correlation}=\frac{\sum_{i,j}(i-\mu_g)(j-\mu_g)\hat P[i][j]}{\sigma_g^2}\in[-1,1],\quad
H_\text{GLCM}=-\!\sum_{i,j:\hat P>0}\hat P[i][j]\log_2 \hat P[i][j].
$$

Intuition: **contrast** is large when neighbours differ a lot (rough texture);
**homogeneity** is large when the matrix hugs the diagonal (smooth); **correlation**
measures linear dependence of neighbour gray levels; **entropy** measures disorder.

## 3. The algorithm

```
build_histogram:  for each ROI voxel v:  h[g(v)] += 1
build_glcm:       for each ROI voxel v:
                    for each of 13 directions δ:
                      w = v + δ
                      if w in-grid and w in ROI:
                        P[g(v)][g(w)] += 1 ; P[g(w)][g(v)] += 1   # symmetric
features:         normalize P, read off Haralick scalars; h -> first-order
```

**Complexity.** The GLCM build is $O(n_\text{roi}\cdot 13)$ integer increments —
linear in ROI size, the dominant cost. The feature reduction is $O(N_g^2)$ (tiny,
64 cells for $N_g=8$). Serial depth is $O(n_\text{roi})$; the parallel *work* is
the same but the parallel *depth* collapses to $O(\log \text{contention})$ because
the increments are independent scatters.

**Access pattern.** Each voxel reads itself and up to 13 neighbours (a small
stencil) and scatters into a *tiny* matrix. Arithmetic intensity is low — this is
a memory-and-atomics-bound histogram, not a compute-bound kernel. That shapes the
GPU design: minimize global-atomic contention, not FLOPs.

## 4. The GPU mapping

**Thread-to-data.** One thread per voxel: linear index
`v = blockIdx.x*blockDim.x + threadIdx.x`, recovered to `(x,y,z)` inside the
kernel. Block size **256** (8 warps) hides memory latency and keeps many blocks
resident; grid = `ceil(n_vox / 256)`.

**The contention problem.** If every one of tens of thousands of threads did
`atomicAdd(&global_P[cell], 1)`, they would serialize on the same ~64 cells — a
textured ROI still has only $N_g^2$ targets, so global atomics would be the
bottleneck.

**The fix — privatized histogram (shared memory).** Each *block* keeps its own
$N_g\times N_g$ GLCM in **shared memory** (on-chip, ~100× faster than global).
Threads atomic-add into that block-private copy (low contention, fast), and at the
end **one flush per non-empty cell** merges the block copy into global memory.
This is the canonical histogram-privatization pattern (docs/PATTERNS.md §1,
exemplar 11.09):

```
  grid of blocks (voxels)                 global GLCM (Ng x Ng)
  ┌───────── block 0 ─────────┐             ┌───────────────┐
  │ 256 threads               │  atomicAdd  │               │
  │  shared s_glcm[Ng*Ng] ────┼──flush────► │   glcm[i][j]  │
  └───────────────────────────┘             │               │
  ┌───────── block 1 ─────────┐  atomicAdd  │               │
  │  shared s_glcm[Ng*Ng] ────┼──flush────► │               │
  └───────────────────────────┘             └───────────────┘
```

**Memory hierarchy used and why.**
- **Constant memory** for the 13 direction offsets: read-only, read by every
  thread → constant-cache broadcast serves a whole warp in one access.
- **Shared memory** for the block-private GLCM: absorbs the hot atomics on-chip.
- **Global memory** for the volume and the final matrix: streamed once; the flush
  does few atomics (≤ $N_g^2$ per block).
- **Registers** hold each thread's `(x,y,z,gi)` and loop counters.

**No CUDA library is used** for the GLCM: it is a custom scatter. The catalog
mentions CUB block-histogram — that is the same idea packaged; here we hand-roll
it so the *mechanism* (shared privatization + atomics) is visible, not hidden
(CLAUDE.md §6.1.6, "no black boxes"). A production version would additionally
grid a second axis over the 13 directions (or over many ROIs) to saturate a large
GPU; we sum directions in one kernel for clarity.

## 5. Numerical considerations

**Counts are integers.** Both the histogram and the GLCM accumulate with
`atomicAdd` on `unsigned int`. Integer addition is **associative and commutative**,
so the result is **independent of thread ordering** — the GPU counts are
deterministic *and bit-identical* to the serial CPU counts. This is the whole
reason we can verify with an **exact** integer check rather than a fuzzy tolerance
(contrast with float atomics, which reorder and would be irreproducible;
docs/PATTERNS.md §3).

**Feature precision.** The derived features (`log2`, `sqrt`, divisions) run in
**double** on both CPU and GPU, over the *same* integer counts, using the *same*
shared reduction code (`haralick_from_glcm`, `first_order_from_histogram`). Any
residual difference is pure floating-point rounding of identical operations
(~$10^{-15}$); we verify to $10^{-9}$ with margin to spare.

**Edge cases handled.** Empty cells contribute nothing (we skip $0\log 0$);
degenerate constant ROI ($\sigma_g=0$) reports the conventional `correlation = 1`;
a 1-voxel ROI (no neighbour pairs) leaves texture features at their zero defaults;
neighbours off the grid or outside the mask simply do not co-occur.

## 6. How we verify correctness

`src/reference_cpu.cpp` builds the GLCM with an obviously-correct serial triple
loop — no parallelism, no cleverness. `main.cu` runs it and the GPU kernel on the
same ROI and asserts:

1. **`glcm_total` identical** (integer count of all pairs) — the exact structural check.
2. **Every feature within `1e-9`** — the derived-value check.

Because the CPU and GPU share the per-voxel math (`radiomics.h`) and the
count→feature reductions (`reference_cpu.cpp`), the *only* thing being tested is
the GPU's atomic scatter vs. the serial loop. When two independent code paths that
build the matrix differently agree to the last integer, we believe the GPU. A
second, *scientific* sanity check: the sample is engineered as a checkerboard, and
the reported features (high contrast, low homogeneity, slightly negative
correlation) match what a checkerboard *must* produce — validating the meaning,
not just the arithmetic (docs/PATTERNS.md §4).

## 7. Where this sits in the real world

Production radiomics (PyRadiomics, cuRadiomics, PyRadiomics-CUDA, MONAI) goes
further in ways worth knowing:

- **More feature families.** GLRLM (gray-level *run lengths*), GLSZM (*size zones*
  of connected same-level regions), NGTDM (neighbourhood gray-tone *difference*),
  and GLDM. Each is a different histogram/scatter but the same GPU pattern applies.
- **3-D shape & wavelet features.** Surface area, sphericity, compactness from the
  mask; features on wavelet-decomposed sub-bands ("multi-scale radiomics").
- **IBSI standardization.** The Image Biomarker Standardization Initiative pins
  down exact discretization, symmetrization, and formula conventions so different
  tools produce *comparable* numbers. Our formulas follow the standard definitions
  but are not IBSI-benchmark-certified.
- **Real image handling.** DICOM/NIfTI I/O, anisotropic voxel spacing, resampling
  to isotropic, intensity outlier clipping, and per-modality normalization — all
  omitted here for a clean text-grid demo.
- **Scale.** cuRadiomics / PyRadiomics-CUDA report ~143× speedups by parallelizing
  exactly the histogram and co-occurrence steps we build here, across full ~10⁶-
  voxel ROIs and whole cohorts. Our tiny sample is launch-overhead-bound (the
  stderr timing says so); the GPU's edge appears only at realistic ROI sizes.

---

## References

- **Haralick, Shanmugam, Dinstein (1973)**, *Textural Features for Image
  Classification*, IEEE TSMC — the original GLCM and its scalar features.
- **PyRadiomics** — [github.com/AIM-Harvard/pyradiomics](https://github.com/AIM-Harvard/pyradiomics):
  the IBSI-aligned CPU reference; read its feature docs for exact definitions.
- **PyRadiomics-CUDA** — [arxiv.org/abs/2510.02894](https://arxiv.org/abs/2510.02894),
  code [github.com/mis-wut/pyradiomics-CUDA](https://github.com/mis-wut/pyradiomics-CUDA):
  the GPU histogram/GLCM mapping and the 143× result.
- **cuRadiomics** (AAPM proceedings) — CUDA texture/GLCM extraction.
- **MONAI** — [github.com/Project-MONAI/MONAI](https://github.com/Project-MONAI/MONAI):
  radiomics inside an end-to-end imaging-AI pipeline.
- **IBSI** — [theibsi.github.io](https://theibsi.github.io/): the standard that
  makes radiomics features reproducible across tools.
- **NVIDIA CUDA C++ Best Practices Guide**, "Shared Memory" and "Atomics" — the
  privatized-histogram pattern used in `glcm_kernel`.
