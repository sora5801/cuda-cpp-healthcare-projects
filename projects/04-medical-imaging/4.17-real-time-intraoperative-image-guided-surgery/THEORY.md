# THEORY — 4.17 Real-Time Intraoperative / Image-Guided Surgery

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

A neurosurgeon plans a tumour resection on a **pre-operative MRI**: they trace the
tumour, mark safe corridors, and build a 3-D model. In the operating room, that
plan is only useful if it can be drawn **in the right place** on the actual
patient — the navigation screen must show "the tumour boundary is *here*, 4 mm to
your left." But the pre-op MRI lives in the scanner's coordinate frame, and the
patient on the table lives in the tracker's frame. **Image-guided surgery (IGS)**
is the discipline of tying those frames together and keeping them tied as the
operation proceeds.

The bridge is **registration**: find the geometric transform between the pre-op
image and the physical patient. The simplest and most fundamental case — the one
this project builds — is **rigid surface registration**. We have:

- a **pre-operative surface** `P` — points on, say, the skin, skull, or an organ
  surface, extracted from the planning MRI/CT (mesh vertices, a segmented
  contour);
- an **intra-operative surface** `Q` — points on the *same* anatomy measured
  during surgery: swiped with a tracked pointer, reconstructed from ultrasound,
  or captured by a depth/stereo camera.

If the patient has not deformed, `P` and `Q` differ by a **rigid body motion** (a
rotation plus a translation) — the same physical surface, seen in two coordinate
frames. Recover that motion and you can transform the entire pre-op plan into the
patient's frame. That recovery is what **Iterative Closest Point (ICP)** does, and
it is the beating heart of every surface-based surgical navigation system.

> **Scope.** Real IGS is a whole pipeline (§7). This project isolates the rigid
> registration stage: given `P` and `Q`, find `R` and `t`. It is the cleanest,
> most self-contained, and most reusable piece.

## 2. The math

We seek a **rigid transform** `T = (R, t)` — a rotation matrix `R ∈ SO(3)` (so
`RᵀR = I` and `det R = +1`) and a translation `t ∈ ℝ³` (millimetres) — that best
maps the moving cloud `P = {p_i}` onto the fixed cloud `Q = {q_j}`.

"Best" is least squares. If we already knew which fixed point `q_{c(i)}` each
moving point `p_i` should map to (the *correspondence* `c`), the problem is:

$$
\min_{R \in SO(3),\, t}\ \sum_{i=1}^{|P|} \left\lVert R\,p_i + t - q_{c(i)} \right\rVert^2 .
$$

This has a beautiful **closed-form solution** (Arun 1987 / Horn / Kabsch):

1. Centroids: `p̄ = (1/n) Σ p_i`, `q̄ = (1/n) Σ q_{c(i)}`.
2. Centred cross-covariance (a 3×3 matrix):
   $$ H = \sum_i (p_i - \bar p)(q_{c(i)} - \bar q)^\top . $$
3. SVD: `H = U S Vᵀ`.
4. Rotation: `R = V · diag(1, 1, det(V Uᵀ)) · Uᵀ`. The `det` term (either +1 or
   −1) flips the least-significant axis if the raw `V Uᵀ` came out as a
   **reflection** (`det = −1`) instead of a rotation — the standard fix that
   forces `R ∈ SO(3)`.
5. Translation: `t = q̄ − R p̄`.

The catch: we **don't** know the correspondence `c`. ICP handles this by
alternating — guess correspondences from the current transform, solve for the
transform, repeat:

$$
c^{(k)}(i) = \arg\min_j \lVert T^{(k)} p_i - q_j \rVert
\quad\Longrightarrow\quad
T^{(k+1)} = \arg\min_{R,t} \sum_i \lVert R\,(T^{(k)}p_i) + t - q_{c^{(k)}(i)}\rVert^2 .
$$

Each half-step never increases the objective, so ICP **monotonically converges**
— to a *local* minimum (this is why the starting guess matters; see §5).

Our reported quality metric is the **RMS correspondence distance** (mm):

$$
\mathrm{RMS}(T) = \sqrt{\tfrac{1}{|P|} \sum_i \min_j \lVert T p_i - q_j \rVert^2 } .
$$

## 3. The algorithm

```
pre-align: T ← centroid_prealign(P, Q)         # T.t = mean(Q) - mean(P), R = I
repeat K times:
    # (A) CORRESPOND + REDUCE  -- the parallel part
    zero accumulators (sumP, sumQ, sumPQ, count)
    for each moving point p_i:                  # independent across i
        tp   ← T · p_i
        j    ← nearest_index(tp, Q)             # O(|Q|) brute-force search
        accumulate tp, Q[j], and tp·Q[j]ᵀ  into the accumulators
    # (B) ALIGN  -- the tiny serial part (host)
    p̄, q̄  ← sumP/n, sumQ/n
    H      ← sumPQ/n − p̄ q̄ᵀ                    # centred covariance identity
    U,S,V  ← svd3x3(H)
    R      ← V·diag(1,1,det)·Uᵀ  (reflection-guarded)
    t      ← q̄ − R p̄
    T      ← (R,t) ∘ T                          # compose increment
    record RMS(T)
```

**Complexity.** Per iteration:

| Step | Serial cost | Parallel (this project) |
|---|---|---|
| Correspond | `O(\|P\|·\|Q\|)` | **work** `O(\|P\|·\|Q\|)`, **depth** `O(\|Q\|)` (one thread/point, each scans Q) |
| Reduce (covariance) | `O(\|P\|)` | `O(\|P\|)` atomic adds, `O(log)` contention depth |
| Align (3×3 SVD) | `O(1)` | `O(1)` on the host |

The correspondence step dominates and is `O(|P|·|Q|)` — quadratic in cloud size.
For 10⁵–10⁶ points this is the entire runtime, and it is **embarrassingly
parallel** across moving points: each point's nearest-neighbour search is
independent. That is the case for the GPU. (A production system also attacks the
`O(|Q|)` *factor* with a k-d tree; see §7 and Exercise 1.)

**Arithmetic intensity / access pattern.** Each thread streams the entire `Q`
array from global memory once per iteration. Neighbouring threads (points) read
the *same* `Q`, so the L2 cache and coalesced loads help; the cross-product
arithmetic per compared pair is small, so the kernel is memory-bandwidth-leaning
on large `Q`.

## 4. The GPU mapping

**One kernel per iteration: `correspond_accumulate_kernel`.**

- **Thread-to-data mapping.** Thread `i = blockIdx.x·blockDim.x + threadIdx.x`
  owns **moving point `P[i]`**. It transforms `P[i]` by the current transform,
  scans all of `Q` to find the nearest fixed point, and adds that pair's
  contribution to the accumulators. The ragged last block is guarded by
  `if (i >= np) return;`.
- **Launch configuration.** `block = 256` threads (a multiple of the 32-lane warp;
  enough warps to hide the latency of the `Q` scan; keeps many blocks resident).
  `grid = ceil(np / 256)`.
- **Memory hierarchy — deliberate choices:**
  - **Constant memory** holds the current rigid transform `c_transform`. Every
    thread reads it to move its point, and it never changes during a launch — the
    exact use case for constant memory's broadcast cache (one fetch serves a whole
    warp). Same idea as the constant-memory query in flagship **1.12**.
  - **Global memory** holds `P`, `Q`, and the 16 accumulators. `P` and `Q` are
    uploaded once and stay resident across all iterations (only the transform and
    the zeroed accumulators change per iteration).
  - **Registers** hold the per-thread running search (`best`, `best_d`, the
    transformed point) — the hot inner loop touches no shared/global scratch.
  - **No shared memory** is used: the win here is the massive thread parallelism
    over the `O(|P||Q|)` search, not intra-block tiling. (A blocked/tiled variant
    that stages chunks of `Q` into shared memory is a natural optimization — noted
    as an exercise.)
- **The reduction** uses `atomicAdd` into 16 global accumulators (3 for `sumP`, 3
  for `sumQ`, 9 for the `sumPQ` outer products, 1 count). See §5 for why they are
  integers.

```
grid  ────────────────────────────────────────────────
 block 0        block 1              block ceil(np/256)-1
[t0 t1 … t255] [t0 … t255]   …      [t0 … t?  (guarded)]
   │  │           │
   ▼  ▼           ▼
 P[0] P[1] …    P[256] …            each thread:
   │                                  tp = R·P[i]+t         (constant mem)
   │  scan all of Q  ───────────────► j = argmin‖tp−Q[j]‖   (global mem, O(|Q|))
   │                                  atomicAdd( pair(tp,Q[j]) → accumulators )
   ▼
 16 int64 accumulators  ──D2H──►  host: 3×3 SVD → (R,t) → compose → next iter
```

**The 3×3 SVD (align step)** runs **on the host**, once per iteration, on the
tiny reduced `H`. We hand-roll a one-sided **Jacobi SVD** (a handful of column
rotations) rather than call a library: at 3×3 it is a few dozen flops, it is
trivially clear, and — crucially — running the *identical host code* on both the
CPU-reference and GPU paths makes the recovered transform bit-identical. **No
black box:** for a *large* least-squares/eigen solve you would call **cuSOLVER**
(`gesvd`/`syevd`) or form the normal equations with **cuBLAS** `gemm` (the
catalog's "cuBLAS for ICP normal-equation solve") — see flagship **2.06** for the
cuSOLVER `syevd` pattern. Here the problem is small enough that the library would
be pure overhead and would obscure the teaching.

## 5. Numerical considerations

- **Precision.** Points are stored `float` (mm; trackers are ~0.1 mm accurate, so
  `float`'s ~7 digits are ample). All *accumulation and the SVD* are `double`, and
  `R`, `t` are `double`, so the transform composed over a dozen iterations does not
  drift.
- **Determinism — the load-bearing trick.** The align step needs three sums over
  all pairs (two centroids and the 3×3 cross-covariance). On the GPU those sums
  are built by many threads doing `atomicAdd`. **Floating-point addition is not
  associative**, and atomic order is nondeterministic, so a *float* atomic sum
  would vary run to run and drift from the CPU. We instead **quantize each term to
  a fixed-point integer** (scale `2¹⁶`) and `atomicAdd` in **64-bit integers**.
  Integer addition commutes, so the reduction is **order-independent →
  reproducible**, and — because the CPU reference quantizes with the identical
  `to_fixed` helper — the GPU sums equal the CPU sums **exactly**. This is the same
  idea as flagship **5.01** (integer energy quanta) and **11.09** (fixed-point
  coordinate sums). The signed 64-bit atomic is synthesized by reinterpreting the
  accumulator as `unsigned long long` (two's-complement addition is bit-identical
  for signed/unsigned).
- **Fixed-point range.** Coordinates span a few hundred mm; a covariance term is
  `≲10⁵` mm², summed over `≲10⁵` points `≲10¹⁰`, and with scale `2¹⁶` a scaled term
  stays `≲10¹⁵` — comfortably under `int64`'s `≈9.2×10¹⁸` ceiling. Quantization
  keeps ~4–5 fractional digits of a millimetre, far finer than any tracker, so the
  recovered transform is physically exact.
- **Local minima & the pre-alignment.** ICP only converges to a **local** optimum;
  from a bad start the first correspondences are wrong and it stalls. On our sample,
  starting from raw identity stalls at **RMS ≈ 3.6 mm**; starting from a **centroid
  pre-alignment** (cancel the gross translation first) converges to **≈ 0.24 mm**
  (the noise floor) in one iteration. This is a real IGS lesson: always coarse-align
  first (centroid match, or a landmark/feature pre-registration). See
  `centroid_prealign` in `icp.h`.
- **Degenerate geometry.** A *flat* fixed surface leaves rotation about its normal
  unconstrained (the covariance is rank-deficient). Our sample is a curved patch (a
  Gaussian bump) precisely so all three rotational axes are observable. The SVD's
  reflection guard also handles the near-degenerate `det = −1` case.

## 6. How we verify correctness

Three independent checks, in increasing strength:

1. **GPU == CPU transform.** `src/reference_cpu.cpp` runs an obviously-correct
   serial ICP; `main.cu` compares the recovered `[R | t]` to the GPU's. Because the
   reduction is integer fixed-point and the SVD is the *same host code*, the
   transforms agree to `1e-9` — in practice the printed `max transform diff` is
   **`0.000e+00`** (bit-for-bit). The tolerance is documented as `1e-9` purely as
   defensive slack. Two independent code paths agreeing exactly is strong evidence
   the implementation is right (docs/PATTERNS.md §4: "exact" tolerance class).
2. **Matching convergence curves.** The per-iteration RMS histories from both paths
   are compared (`max history diff`, also `0.0`).
3. **Science-level check.** The recovered rotation is the **inverse (transpose)** of
   the known ground-truth rotation the synthetic sample was built with, and the
   final RMS falls to the **injected noise floor** (~0.24 mm for 0.15 mm/axis noise
   over 3 axes) — so ICP recovered the *true* alignment, not just some CPU==GPU
   fixed point.

## 7. Where this sits in the real world

This project is the **rigid surface-registration** stage of IGS. Production systems
add, around it:

- **A fast correspondence structure.** Real ICP never brute-forces the nearest
  neighbour; it uses a **k-d tree**, octree, or GPU spatial hash to make each query
  `O(log|Q|)`. (Our `O(|Q|)` scan is for clarity — Exercise 1.)
- **Better error metrics & robustness.** **Point-to-plane** ICP (minimize distance
  to the local tangent plane) converges far faster on smooth surfaces; **trimmed /
  robust** ICP rejects outliers from occlusion and partial overlap.
- **Deformable registration.** Tissue is not rigid — the brain **shifts** several
  millimetres once the skull is open, lungs and abdomen breathe. Beyond ICP, systems
  run **Demons** (a diffusion-based deformable image registration; the catalog's
  "GPU Demons for brain-shift") or **biomechanical FEM** driven by intra-op
  ultrasound. Rigid ICP is the initialization those methods build on.
- **The rest of the pipeline.** Intra-op **CBCT reconstruction** (FDK — this repo's
  flagship **4.01**), **DRR** generation for 2D/3D X-ray registration, CNN
  **instrument segmentation** (U-Net/YOLO via cuDNN/TensorRT), **Kalman**-filtered
  tool tracking, and **OpenGL/Vulkan interop** to overlay it all on live video at
  <10 ms latency (NVIDIA **Holoscan**, **3D Slicer** + OpenIGTLink, **PLUS**).

The align step's linear algebra is what the catalog means by "cuBLAS for ICP
normal-equation solve": at scale you form and solve the fit with cuBLAS/cuSOLVER
rather than a 3×3 hand-roll. The pattern generalizes directly to those libraries
(see flagship **2.06** for cuSOLVER `syevd`).

---

## References

- **P.J. Besl & N.D. McKay (1992)**, "A Method for Registration of 3-D Shapes",
  *IEEE TPAMI* — the original ICP algorithm.
- **K.S. Arun, T.S. Huang, S.D. Blostein (1987)**, "Least-Squares Fitting of Two
  3-D Point Sets", *IEEE TPAMI* — the SVD closed-form rigid fit used in the align
  step. (Horn 1987 gives the quaternion equivalent.)
- **S. Rusinkiewicz & M. Levoy (2001)**, "Efficient Variants of the ICP Algorithm",
  *3DIM* — point-to-plane, sampling, and robustness variants (Exercises 2–3).
- **3D Slicer / OpenIGTLink** — <https://github.com/Slicer/Slicer> — the open IGS
  reference platform; learn how tracker transforms stream into a navigation scene.
- **PLUS toolkit** — <https://github.com/PlusToolkit/PlusLib> — real-time US
  acquisition/reconstruction feeding the intra-op surface.
- **NVIDIA Holoscan SDK** — <https://github.com/nvidia-holoscan/holoscan-sdk> — the
  low-latency GPU streaming pipeline for surgical video/sensors.
- **RTK** — <https://github.com/RTKConsortium/RTK> — intra-op CBCT (FDK) recon (cf.
  flagship 4.01).
- **cuSOLVER / cuBLAS docs** — the library path for the align solve at scale
  (cf. this repo's flagship 2.06).
