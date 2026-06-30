# THEORY — 2.4 Cryo-ET Subtomogram Averaging (reduced-scope teaching version)

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See
> [README.md](README.md) for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

**Cryo-electron tomography (cryo-ET)** flash-freezes a whole cell (or a thin
lamella of one) and images it in a transmission electron microscope while
tilting the sample through a range of angles (typically −60° to +60°).
Back-projecting that *tilt series* reconstructs a 3-D **tomogram**: a noisy
volume showing the cell's molecular landscape *in situ* — ribosomes translating,
proteasomes degrading, membranes curving — at nanometer scale.

The catch is signal-to-noise. To avoid destroying the radiation-sensitive
sample, each image uses a tiny electron dose, so a single tomogram is extremely
noisy and any one copy of a molecule is barely visible. But a cell contains
**thousands of copies of the same machine**, each at a random orientation.
**Subtomogram averaging (STA)** exploits this redundancy:

1. **Pick** small cubes ("subtomograms") around each copy of the target.
2. **Align** every subtomogram to a common reference (find its rotation and
   translation).
3. **Average** the aligned cubes. Random noise cancels as `1/√N`; the coherent
   signal reinforces. With thousands of particles you recover a clean structure,
   sometimes to sub-nanometer resolution — *from data in which no single particle
   was interpretable*.

This project teaches the **align + average** inner loop, where the GPU earns its
keep. The alignment step is a **cross-correlation search**, and that is exactly
what cuFFT accelerates.

### The "missing wedge"

Because the stage can only tilt to ~±60° (not ±90°), a wedge of directions in
Fourier space is never measured. This **missing wedge** smears densities along
the beam axis and is *the* characteristic cryo-ET artifact; neural methods like
IsoNet try to inpaint it. We do not model the wedge in the demo (our synthetic
motif is fully sampled), but we describe it here and in §7 because no STA
discussion is complete without it.

## 2. The math

A subtomogram is a real-valued function `g(v)` on a `d×d×d` voxel grid `v ∈ ℤ³`.
The reference is `f(v)`. After **zero-meaning** both (subtract the mean so a
constant offset cannot dominate), their similarity at a translational shift
`s ∈ ℤ³` is the **cross-correlation**

```
    (f ⋆ g)(s) = Σ_v  f(v) · g(v + s).
```

To make scores comparable across particles of different contrast we use the
**normalized cross-correlation (NCC)** — the Pearson correlation coefficient:

```
    NCC(f, g; s) =  Σ_v f(v) g(v+s)  /  sqrt( Σ_v f(v)²  ·  Σ_v g(v)² ).
```

`NCC ∈ [−1, 1]`: it is `1` for identical shapes, `0` for unrelated ones. The
denominator is the product of the two volumes' **energies** (sums of squares).

**Symbols.** `d` = cube edge in voxels; `V = d³` = voxels per cube; `n_sub` =
candidates; `A = n_angles` = trial angles; `θ_k = 2πk/A` (radians) = the k-th
trial rotation about the z-axis; intensities are arbitrary (synthetic) units.

**Alignment** = find the rotation `R` and shift `s` that maximize
`NCC(f, R·g; s)`. We discretize the rotation search to `A` trial angles and, for
each, take the **best shift** via the correlation peak. The winning `(R, s)` is
the particle's pose.

### The cross-correlation theorem (the whole trick)

Computing `(f ⋆ g)(s)` directly costs `O(V)` per shift, and there are `V` shifts,
so `O(V²)` per (particle, angle). The **cross-correlation theorem** turns the
entire shift search into three FFTs:

```
    f ⋆ g  =  IFFT(  conj(FFT(f)) · FFT(g)  ).
```

The output gives the correlation at **all shifts at once**. With the FFT costing
`O(V log V)`, one (particle, angle) job drops from `O(V²)` to `O(V log V)`. For
`d = 16`, `V = 4096` (modest), but the gap widens fast: at `d = 64`,
`V ≈ 2.6·10⁵`, direct is `~7·10¹⁰` while FFT is `~4.6·10⁶`. The `conj` (complex
conjugate of the reference spectrum) is precisely what distinguishes
**correlation** (a match score over shifts) from **convolution** (a filter). The
peak of the IFFT is the best translational alignment; the value at voxel
`(0,0,0)` is the score at **zero shift** — our verification hook.

## 3. The algorithm

For each candidate `s` and each trial angle `k`:

1. **Rotate** the candidate by `θ_k` about z (bilinear, backward map) → `g_rot`.
2. **FFT** both `g_rot` and the reference `f` (real→complex, `R2C`).
3. **Multiply** per frequency: `H = conj(F) · G` (the theorem).
4. **Inverse-FFT** `H` (complex→real, `C2R`) → the correlation field over all
   shifts.
5. **Reduce**: the field's **peak** is the best-shift correlation; the value at
   `(0,0,0)` is the **zero-shift** correlation. Normalize both by
   `sqrt(energy(f)·energy(g_rot))` → NCC.

Then per candidate, pick the **angle with the highest peak NCC** = its pose, and
**average** all candidates rotated to their poses → the refined reference.

**Complexity** (n candidates, A angles, V = d³ voxels):

| Step | Work | Notes |
|------|------|-------|
| Rotation | `O(n·A·V)` | one interpolation per output voxel |
| FFT (fwd+inv) | `O(n·A·V log V)` | batched over all `n·A` jobs |
| Spectral multiply | `O(n·A·V)` | one complex mul per frequency |
| Reduction | `O(n·A·V)` | peak + zero-shift per job |

The FFT dominates and is exactly the part cuFFT parallelizes. The **direct** CPU
reference instead computes only the **zero-shift** NCC the definition way
(`O(n·A·V)` total) — cheap *because* it skips the full shift search, which keeps
the baseline short and obviously correct.

## 4. The GPU mapping

This is a **"use a CUDA library" + batched independent jobs** project
(`docs/PATTERNS.md` §1, §5; the exemplar is flagship `8.03`, which also uses
cuFFT). Every `(candidate, angle)` pair is an independent job; we lay out `n·A`
jobs and process them as one batch.

**Thread / grid layout**

- `rotate_kernel`: one thread per **output voxel** of one job. The grid is 3-D:
  `grid.x` covers the `V` voxels, `grid.y = A`, `grid.z = n_sub`. Thread
  → job `j = cand·A + angle` and voxel index in the cube. The kernel recomputes
  `θ_k` with the **same formula** as the host (`trial_angle`) so the two cannot
  drift.
- `xcorr_mul_kernel`: one thread per `(job, frequency-bin)`; reads the single
  shared reference spectrum and multiplies in place.
- `energy_kernel` / `reduce_kernel`: **one block per job**, a deterministic
  shared-memory **tree reduction** (sum of squares; max + read-`[0]`).

```
            grid.z = n_sub (candidates)
              │
              ▼
   ┌───────────────────────────┐   each (cand, angle) = one independent JOB
   │ job 0   job 1   …   job J-1│   J = n_sub * n_angles
   └───────────────────────────┘
        │ rotate_kernel  (1 thread / output voxel)
        ▼
   [ n_sub*A rotated cubes ]  ──cuFFT R2C (batched)──▶  [ spectra ]
        │ xcorr_mul_kernel  (conj(F)·G per frequency)
        ▼
   [ products ]  ──cuFFT C2R (batched, ÷V)──▶  [ correlation fields ]
        │ reduce_kernel  (1 block / job: peak + corr[0], ÷ energies)
        ▼
   [ (peak NCC, zero-shift NCC) per job ]  ──host argmax over angles──▶  poses
```

**Which library does what.** `cufftPlanMany(..., CUFFT_R2C, n_jobs)` builds one
plan that batches `n_jobs` 3-D real→complex FFTs laid out contiguously (input
stride `V`, output stride `nfreq = d·d·(d/2+1)`). A real signal's FFT is
Hermitian-symmetric, so `R2C` stores only the non-redundant half (last axis
`d/2+1`) — half the memory and work. `CUFFT_C2R` is the inverse. Hand-rolling a
batched, multi-dimensional, mixed-radix FFT correctly *and* fast is weeks of
work; cuFFT is the right tool, kept from being a black box by documenting its
inputs, layout, and exact transform in `kernels.cu` (CLAUDE.md §6.1.6).

**Memory hierarchy.** Cubes live in **global memory** (too big for shared
memory). The reference spectrum is read by every job — a candidate for
**constant/texture** memory at scale, but at `d=16` it is tiny and L2 handles the
reuse, so we keep the code simple and say so. Reductions stage partials in
**shared memory**. No atomics anywhere (see §5).

**Why batching matters.** One `cufftExecR2C` over `n·A` cubes saturates the GPU
far better than `n·A` separate small FFTs: plan setup is amortized and the
scheduler has thousands of independent transforms to overlap. This mirrors how
real STA batches particles per GPU.

## 5. Numerical considerations

- **Precision.** Single precision (FP32) end to end: cuFFT's mainstream path and
  what STA packages use for correlation (resolution is limited by the data, not
  by FP64). The CPU reference accumulates the zero-shift dot product in FP32 too,
  so the comparison is apples-to-apples.
- **cuFFT normalization.** The `R2C`→`C2R` round trip is **unnormalized**: it
  scales the result by `V`. We divide by `V` (`invV`) in `reduce_kernel` to
  recover the true correlation sum. Forgetting this `1/V` is the classic FFT bug.
- **Correlation vs convolution.** Using `conj(F)·G` (not `F·G`) is what makes the
  peak land at the true shift. Drop the conjugate and you compute convolution and
  the peak is mirrored — a good thing to break on purpose to learn.
- **Determinism (PATTERNS.md §3).** Every reduction is an **index-ordered tree**
  in shared memory — never `atomicAdd` into a float accumulator, whose
  non-associative ordering would make sums (and peak ties) run-dependent. The
  argmax-over-angles uses strict `>` so ties resolve to the lowest index on both
  CPU and GPU. Result: stdout is **byte-identical** every run.
- **Zero-variance guard.** If a cube has no variance the NCC denominator is `0`;
  we return `0` rather than divide by zero.

## 6. How we verify correctness

Two independent checks, both in `main.cu`:

1. **The FFT identity.** The GPU's **zero-shift** NCC (value at IFFT voxel
   `(0,0,0)`, normalized) must equal the CPU's **direct** zero-shift NCC (a
   literal `Σ f·g` over the same rotated cube). This proves the whole FFT
   pipeline computes the correlation it claims to. Measured agreement on the
   sample: **worst error ≈ 3.2 × 10⁻⁷** — far inside the documented `1 × 10⁻³`
   ceiling. The tolerance is roomy on purpose: the FFT route rounds differently
   from the direct sum (a real effect that grows ~`√log V`), and we want the test
   robust across GPUs/arches, not brittle (PATTERNS.md §4).
2. **Pose agreement + planted ground truth.** Both paths must select the **same
   best angle** for every candidate. Better still, the synthetic data is built so
   the correct answer is **known**: candidate `s` is the motif rotated by a
   planted angle, so the alignment search *should* recover trial indices
   `[1, 3, 5, 7, 9, 11]` — and it does, at peak NCC ≈ 0.965 (PATTERNS.md §6). That
   validates the *science* (the right poses were found), not just CPU==GPU
   agreement.

The refined-average **core intensity** (`mean|voxel|`) is reported as a single
deterministic scalar; CPU and GPU produce the identical `0.097395` because they
average the same rotated cubes.

Why an independent serial baseline is convincing: it shares no code path with the
GPU (no FFT, no kernels, plain loops), so agreement is unlikely to be a shared
bug — it is two roads to the same number.

## 7. Where this sits in the real world

Our demo is a **deliberately reduced-scope teaching version** (CLAUDE.md §13).
The real algorithm, and how production tools differ:

| Aspect | This demo | Production STA (RELION-4, Dynamo, emClarity, STOPGAP) |
|--------|-----------|--------------------------------------------------------|
| Orientation search | discrete **in-plane** (1 angle about z) | full **3-D** search over 3 Euler angles, refined locally |
| Translational search | full, via FFT peak | same idea, FFT-based, plus sub-voxel peak fitting |
| Missing wedge | not modeled | **wedge mask** in Fourier space; constrained cross-correlation; IsoNet CNN to inpaint |
| Weighting | none | **CTF** correction, dose/exposure weighting, Bayesian priors (RELION), band-pass filters |
| Iteration | single pass | iterate align→average→re-align until convergence; gold-standard FSC for resolution |
| Classification | none | 3-D classification to separate conformational/compositional states |
| Scale | 6 cubes, `16³` | 10³–10⁶ particles, `64³`–`128³`, multi-GPU |

The **load-bearing idea is identical**: alignment is an FFT-accelerated
cross-correlation search, and averaging beats down noise by `1/√N`. Everything
above is engineering and statistics layered on that core — the core this project
teaches.

**Reconstruction (the other GPU hot spot).** The catalog also lists weighted
back-projection (WBP) and SART for building the tomogram from the tilt series.
That is the same *gather / back-projection* pattern as CT (flagship `4.01`); we
focus on the alignment half because it showcases the cross-correlation theorem
and cuFFT.

---

## References

- **RELION-4** — Bayesian subtomogram averaging with CUDA
  (`github.com/3dem/relion`). Study its Fourier-space alignment and CTF model.
- **Dynamo** — MATLAB/GPU STA toolbox
  (`wiki.dynamo.biozentrum.unibas.ch`). Readable docs on alignment projects and
  table conventions.
- **IsoNet** — deep-learning missing-wedge correction
  (`github.com/IsoNet-cryoET/IsoNet`). How a CNN inpaints the wedge.
- **IMOD** — tomogram reconstruction toolkit (`bio3d.colorado.edu/imod`). The WBP
  side this demo does not cover.
- Bharat & Scheres, *Nat. Protoc.* 11, 2054 (2016) — a clear STA-in-RELION
  walkthrough; good for the iterate-to-convergence picture.
- cuFFT documentation (`docs.nvidia.com/cuda/cufft`) — `cufftPlanMany`, R2C/C2R
  layouts, and the unnormalized-inverse caveat we handle in §5.
