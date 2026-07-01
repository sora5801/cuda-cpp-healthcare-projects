# THEORY — 4.32 GPU-Accelerated Landmark Detection

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See
> [README.md](README.md) for the quick tour and build steps.
>
> _Educational only — not for clinical use. All data here is synthetic._

---

## 1. The science

**Anatomical landmark detection** localizes clinically meaningful *points* in a
medical image — a vertebral endplate, a femoral head centre, a dental cusp, a
cephalometric point. These points drive downstream tasks: initializing image
registration, taking measurements (Cobb angle, leg-length), and planning surgery
or radiotherapy. A radiologist can click a dozen points in a 2D X-ray in
seconds; doing it reproducibly for **dozens of landmarks across a full 3D CT**,
for thousands of scans, needs automation.

The dominant modern approach is **heatmap regression**. Instead of asking a
network to output raw coordinates (hard to train — a single number with no
spatial structure), we ask it to output, *for each landmark `l`*, an entire 3D
volume `H_l[z,y,x]`, a "probability blob" that is high near the landmark and low
elsewhere. The network is trained so `H_l` matches a **Gaussian target** centred
on the annotated point. Architectures: the **stacked hourglass** (repeated
encode→decode with skip connections) and the **3D U-Net**.

Why the GPU is not optional: for **100 landmarks in a 512³ CT**, the output
tensor is `100 × 512³ ≈ 1.3 × 10¹⁰` voxels ≈ **13 GB** of activations — training
and even inference are infeasible on a CPU at clinical speeds. (An alternative
family, **multi-agent reinforcement learning**, has one agent per landmark
*navigate* the volume; the GPU then runs all agents in parallel.)

### What this project implements (and what it deliberately does not)

Training a 3D U-Net is out of scope for a single self-contained CUDA teaching
project (it would pull in cuDNN, a data pipeline, and hours of GPU time). We
implement the **inference-time DECODE step** that *every* heatmap method ends
with, and which is itself a clean, instructive GPU problem: **given the predicted
heatmaps, extract each landmark's coordinate.** Two classic decoders:

1. **Argmax** — the integer voxel with the largest value (coarse, exact).
2. **Soft-argmax** — the intensity-weighted centroid of a small window around
   that peak (sub-voxel: recovers fractional positions the grid cannot store).
   This is what production toolkits use for final accuracy.

The heatmaps themselves are supplied as a **synthetic** dataset of Gaussian blobs
at known centres (see `data/README.md`), so we can measure how well the decode
recovers the planted points. This is a reduced-scope, honestly-labeled teaching
version per CLAUDE.md §13; §7 below describes the full pipeline.

## 2. The math

**Inputs.** `L` heatmaps over a shared voxel grid of size `nx × ny × nz`. Write
`V = nx·ny·nz` voxels per heatmap; heatmap `l` is `H_l : {0..nx-1}×{0..ny-1}×{0..nz-1} → [0,1]`.

**Argmax (coarse localization).** The peak voxel of landmark `l`:

```
(p_x, p_y, p_z) = argmax_{(x,y,z)} H_l(x,y,z)
```

Ties (several voxels with the exact same value) are broken by **lowest
row-major flat index** — a deterministic rule both CPU and GPU obey.

**Soft-argmax (sub-voxel refinement).** Over a cube window `W` of half-width `R`
around the peak, the intensity-weighted centroid:

```
                Σ_{v ∈ W} w(v) · pos(v)
coord_l  =  ───────────────────────────────,     w(v) = clamp(H_l(v), 0, 1)
                Σ_{v ∈ W} w(v)
```

computed independently per axis (`pos(v)` = that voxel's x, y, or z). The window
restricts the average to the blob, so distant noise cannot pull the centroid.

**Recovery error (science check).** With a synthetic ground-truth centre
`c_l = (c_x, c_y, c_z)`, the decode error is the Euclidean distance
`‖coord_l − c_l‖₂`, reported in voxels.

## 3. The algorithm

Per landmark:

```
1. ARGMAX:      scan all V voxels, track (best value, best flat index).      O(V)
2. UNFLATTEN:   convert the winning flat index to (p_x, p_y, p_z).           O(1)
3. SOFT-ARGMAX: over the (2R+1)³ window, accumulate  Σw, Σw·x, Σw·y, Σw·z.   O(R³)
4. FINALIZE:    coord = (Σw·x, Σw·y, Σw·z) / Σw   (per axis).                O(1)
```

**Complexity.** Serial cost is `O(L·V)` — dominated by step 1, the full-volume
scan (`R³` is a tiny constant, 125 for `R=2`). For 100 landmarks over 512³ that
is `1.3 × 10¹⁰` voxel reads: perfectly parallel across the `L` landmarks *and*
across the `V` voxels, which is exactly why it belongs on the GPU.

**Data-access pattern.** Step 1 is a **streaming reduction** (read each voxel
once, low arithmetic intensity → memory-bandwidth bound). Step 3 touches only a
tiny local window. So the GPU decode is essentially "read the volume once as fast
as memory allows, then do O(1) math."

## 4. The GPU mapping

The pattern (docs/PATTERNS.md): **one independent reduction per landmark** — a
blend of "score N independent items" (`1.12`) and "parallel assign + atomic
reduce" (`11.09`). Decoding landmark `l` depends only on volume `l`, so:

```
grid  : L blocks          blockIdx.x = l   -> the landmark this block decodes
block : 256 threads       threadIdx.x = t  -> a lane striding over voxels
```

Within a block the 256 threads cooperate on one heatmap, in two phases:

```
 Volume l (V voxels)                     one THREAD BLOCK (256 lanes)
 ┌───────────────────────────┐           ┌───────────────────────────────┐
 │  ....  strided over lanes  │  PHASE 1  │ each lane: best (val, idx)     │
 │  t t+256 t+512 ...         │  ───────► │ tree-reduce in __shared__      │
 └───────────────────────────┘           │   -> lane 0 holds the argmax   │
                                          ├───────────────────────────────┤
 (2R+1)³ window around peak     PHASE 2   │ lanes stride the window,       │
                               ───────►   │ atomicAdd fixed-point weights  │
                                          │ into __shared__ 64-bit sums    │
                                          │   -> lane 0 divides & writes   │
                                          └───────────────────────────────┘
```

**Phase 1 — parallel argmax.** Each lane scans voxels `t, t+B, t+2B, …`,
tracking its own best `(value, flat-index)`. Partials go to `__shared__` arrays
`s_val[256], s_idx[256]`; a **tree reduction** (halve the active range each step,
`__syncthreads()` between steps) collapses them to lane 0's single winner. The
merge rule — *larger value wins; on an exact tie, lower index wins* — reproduces
the CPU's row-major tie-break so the two agree bit-for-bit.

**Phase 2 — parallel soft-argmax.** The `(2R+1)³` window is flattened to
`[0, WIN)` and striped across lanes. Each contributing lane `atomicAdd`s its
weight into **block-scoped shared** accumulators. Shared-memory atomics are used
(not global) because every contributor for a landmark lives in one block, and
shared atomics are far cheaper.

**Memory hierarchy & why.**
- **Global memory:** the `L·V` heatmap floats. Phase 1's coalesced streaming read
  is the bandwidth bottleneck; this is the step that scales.
- **Shared memory:** the argmax partials and the soft-argmax accumulators —
  block-private, fast, freed when the block ends.
- **Registers:** each lane's running `(best, best_i)` and loop indices.

**No CUDA library is needed here.** The decode is a custom reduction, so we
hand-write it (and teach it). In the *full* pipeline (§7) the heavy lifting —
the 3D convolutions of the U-Net — is exactly where **cuDNN** and **Tensor
Cores** dominate, and a regression head would use **cuBLAS** GEMM.

## 5. Numerical considerations

**Precision.** Heatmap values are FP32 (that is what a network emits). The
argmax compares floats directly — no arithmetic, so no rounding issue; the only
subtlety is ties, handled by the index rule above.

**Determinism — the key teaching point.** The soft-argmax is a sum over the
window, and on the GPU many threads add into the *same* accumulators
concurrently. Floating-point addition is **not associative**, so a naive *float*
`atomicAdd` sum would depend on the (nondeterministic) order threads finish → the
last bits would wiggle run-to-run and never exactly match the CPU. The fix
(docs/PATTERNS.md §3): **quantize each weight to a 64-bit integer**
(`w = ⌊H·10⁶⌋`, `landmark.h::quantize_weight`) and accumulate in integers.
Integer addition commutes, so the totals are **order-independent, reproducible,
and bit-identical to the CPU's serial integer sums**. The single floating-point
operation left is the final `Σw·x / Σw` division, done once (identically) on both
sides via the shared `finalize_softargmax`.

**Race conditions.** The only concurrent writes are the phase-2 `atomicAdd`s,
which are correct by construction; the tree reduction is race-free thanks to the
`__syncthreads()` barriers. Overflow is impossible for realistic sizes:
`10⁶ · V ≈ 1.3 × 10¹⁴ ≪ 9.2 × 10¹⁸ = INT64_MAX`.

## 6. How we verify correctness

`src/reference_cpu.cpp` is an independent, obviously-correct serial decoder:
nested loops for the argmax, a serial window sum for the soft-argmax. `main.cu`
runs both and checks two things:

- **Integer argmax peaks match EXACTLY** (`==`). Same values, same tie-break.
- **Sub-voxel coordinates match within `1e-9`.** Because the integer weight sums
  are bit-identical, only the final `double` division can differ, and only in its
  last ulps. `1e-9` is far below any voxel spacing yet safely above that noise
  (docs/PATTERNS.md §4). In practice the observed max difference is `0.0`.

The convincing part: the CPU and GPU codes share only the tiny per-voxel helpers
in `landmark.h`; the *control flow* (serial nested loops vs. block-parallel
reduction with atomics) is completely different. Agreement between two such
different implementations is strong evidence the parallel one is correct.

A **second, stronger check** validates the *science*, not just CPU==GPU: the
recovered coordinate is compared to the *planted* Gaussian centre, and the demo
reports the worst recovery error (~0.2 voxels here). The residual is honest — a
finite window plus Gaussian tails bias the centroid slightly; a larger radius or
a parabolic sub-voxel fit would shrink it (see Exercises).

## 7. Where this sits in the real world

Production landmark detectors do far more than the decode:

- **The network.** [MONAI](https://github.com/Project-MONAI/MONAI) and
  [nnDetection](https://github.com/MIC-DKFZ/nnDetection) train 3D U-Nets /
  hourglasses whose 3D convolutions run on **cuDNN** + **Tensor Cores**, with
  GPU-resident data augmentation (elastic deformation, Gaussian blur). The
  heatmaps this project *consumes* are that network's *output*.
- **Coarse-to-fine cascades.** Detect landmarks at low resolution, then crop and
  refine — cutting the 13 GB tensor problem down to tractable patches.
- **Anatomy-guided priors.** Enforce that vertebrae are ordered along the spine,
  rejecting geometrically impossible detections.
- **Reinforcement-learning detectors (MARL-DQN).** Each landmark is an agent that
  *walks* toward its target; the GPU runs all agents in parallel. No dense
  heatmap tensor at all.

Our contribution is the **decode kernel** that turns any of those methods'
heatmaps into coordinates — deterministically, verifiably, and fast.

---

## References

- **Newell, Yang, Deng (2016), "Stacked Hourglass Networks."** The heatmap-
  regression architecture; read for *why* heatmaps beat direct coordinate
  regression.
- **Payer et al. (2019), "Integrating spatial configuration into heatmap
  regression."** Soft-argmax and spatial priors for landmarks — the refinement
  step we implement, in its trained form.
- **MONAI** — <https://github.com/Project-MONAI/MONAI> — production transforms
  for landmark heatmaps; study `KeypointsToHeatmap` / decode utilities.
- **nnDetection** — <https://github.com/MIC-DKFZ/nnDetection> — a self-configuring
  GPU detection framework; the "how to scale to real 3D data" reference.
- **VerSe** — <https://github.com/anjany/verse> — the vertebral-landmark
  benchmark; where real annotated heatmaps come from.
