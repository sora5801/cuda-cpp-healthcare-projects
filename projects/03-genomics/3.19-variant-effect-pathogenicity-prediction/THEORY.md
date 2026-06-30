# THEORY — 3.19 Variant Effect / Pathogenicity Prediction

> The deep didactic companion to this project. Read the [README](README.md) first
> for orientation, then this for the *why*. Code references point at
> [`src/vep_model.h`](src/vep_model.h) (the shared math),
> [`src/kernels.cu`](src/kernels.cu) (the GPU path), and
> [`src/reference_cpu.cpp`](src/reference_cpu.cpp) (the trusted baseline).
>
> _Educational only — not for clinical use._

---

## 1. The science

A genome is ~3.2 billion DNA bases (A, C, G, T). Between any two people millions of
positions differ. Most differences are harmless; a few break a gene and cause disease.
The central question of clinical genomics is: **given a variant, is it pathogenic or
benign?**

Three families of methods answer it, in rough historical order:

1. **Conservation scores** (SIFT, PolyPhen, GERP). Intuition: if a position has been
   the same across hundreds of millions of years of evolution, a change there is
   probably bad. These are alignment statistics — fast but shallow.
2. **Deep mutational scanning (DMS)** experiments measure, in the lab, the functional
   effect of *every* amino-acid substitution in a protein. Gold-standard but expensive
   and protein-by-protein (catalogued in MaveDB).
3. **Deep learning over sequence context.** A neural network reads the DNA (or protein)
   sequence *around* the variant and predicts a functional readout — gene expression
   (Enformer), amino-acid likelihood (ESM-1v), or a calibrated pathogenicity
   (AlphaMissense). The **variant effect** is the network's output on the **alternate**
   allele minus its output on the **reference** allele.

This project models family (3) — the one that needs the GPU. The biology we keep is the
*shape*: a variant sits at the centre of a fixed **context window** of DNA, and its
effect is a difference of two model evaluations. The biology we drop (for teaching) is
the trained network itself, which we replace with a tiny fixed CNN.

> **A motif, concretely.** Real regulatory/coding signals often look like short
> sequence **motifs** (a transcription-factor binding site, a splice donor `GT`). Our
> toy model plants two 5-mers: a "deleterious" `CAGCT` (creating it raises the score)
> and a "protective" `TATAT` (creating it lowers it). This is a caricature of how a
> real CNN filter detects a functional motif.

---

## 2. The math

### 2.1 One-hot encoding

A window of `L` bases is encoded as an `L × 4` one-hot matrix `X`: row `p` is the
column `e_{b_p}` where `b_p ∈ {A,C,G,T}` is the base at position `p`. Exactly one of the
four entries per row is 1. We store only the **index** `b_p ∈ {0,1,2,3}` (an `int8`),
not the full matrix — the one-hot column has a single 1, so any dot product against it
collapses to a single lookup. (`VEP_WINDOW = 21`, so the variant sits at centre index
`10`.)

### 2.2 The toy network `f(X)`

A 1-D convolutional classifier with `K` filters of width `W`:

- **Conv1D + ReLU.** Filter `k` has weights `w_{k,c,j}` (channel `c`, offset `j`) and
  bias `β_k`. Its pre-activation at start position `p` is

  ```
  a_{k,p} = β_k + Σ_{j=0}^{W-1} Σ_{c=0}^{3} w_{k,c,j} · X[p+j, c]
          = β_k + Σ_{j=0}^{W-1} w_{k, b_{p+j}, j}          (one-hot collapse)
  ```

  followed by `h_{k,p} = ReLU(a_{k,p}) = max(a_{k,p}, 0)`.

- **Global max pool.** Per filter, take the strongest response over all positions:
  `g_k = max_p h_{k,p}`. This asks "does motif `k` appear *anywhere* in the window?",
  the standard motif-detector pooling.

- **Dense + sigmoid.** A linear layer collapses the `K`-vector to one logit,
  `z = b + Σ_k v_k g_k`, squashed to a pseudo-probability
  `f(X) = σ(z) = 1/(1+e^{−z}) ∈ (0,1)`.

### 2.3 The variant effect (delta score)

For a variant with reference window `X^{ref}` and alternate window `X^{alt}` (identical
except at the centre base):

```
Δ = f(X^{alt}) − f(X^{ref})        ∈ (−1, 1)
```

`Δ > 0` ⇒ the alternate allele looks **more** pathogenic. This is exactly the "ref/alt
difference" recipe (Enformer) and a sibling of the **log-odds ratio**
`LOR = logit(p_alt) − logit(p_ref)` used by protein language models (ESM-1v) — we report
the probability difference for simplicity; Exercise 3 swaps in the logit form.

---

## 3. The algorithm

```
load N variants (each: ref window + alt window, differing only at the centre)
init fixed model weights (deterministic; planted motifs)
for each variant i in 0..N-1:          # independent — parallel over i
    s_ref = f(ref_window_i)            #   forward pass 1
    s_alt = f(alt_window_i)            #   forward pass 2
    effect[i] = s_alt - s_ref          #   delta score
rank variants by effect (descending)   # "most pathogenic-looking" first
```

**Complexity.** One forward pass is `O(K · (L−W+1) · W)` multiply-adds; here
`8 · 17 · 5 = 680`. Two passes per variant, `N` variants ⇒ **`O(N · K · (L−W+1) · W)`**
total — *linear in the number of variants*. Serial (CPU) and parallel (GPU) do the same
arithmetic; the GPU just runs the `N` variants concurrently instead of one at a time.

The shared per-variant routine is [`vep_variant_effect()`](src/vep_model.h); the CPU
loops it ([`score_variants_cpu`](src/reference_cpu.cpp)) and the GPU runs it
one-per-thread ([`score_variants_kernel`](src/kernels.cu)).

---

## 4. The algorithm on the GPU — the GPU mapping

This is the **independent-jobs · constant-memory** pattern (PATTERNS.md §1, exemplar
`1.12` Tanimoto). The mapping:

```
                          grid (1-D)
   block 0          block 1                  ... (grid-stride loop wraps)
 ┌───────────┐   ┌───────────┐
 │ t0 t1 ...  │   │ t0 t1 ...  │   each THREAD i  -> one VARIANT
 └───────────┘   └───────────┘   reads ref[i], alt[i]; writes effect[i]
       │
       ▼
   c_model  (__constant__)  ── broadcast to every warp lane in one transaction
```

- **Thread → data.** Thread `i = blockIdx.x·blockDim.x + threadIdx.x` scores variant `i`.
  A **grid-stride loop** (`i += blockDim.x·gridDim.x`) lets a capped grid (≤1024 blocks)
  cover an arbitrarily large batch, so the launch config is independent of `N`.
- **Block size.** 256 threads = 8 warps: a multiple of the 32-lane warp, enough warps to
  hide the few global loads, and many blocks resident for occupancy on sm_75…sm_89.
- **Memory hierarchy:**
  - **Constant** — the model `c_model`. Read by every thread, never written, identical all
    launch: the constant cache broadcasts one address warp-wide in a single transaction.
    `VepModel` is a fixed-size POD (a few KB), well within the 64 KB bank; uploaded once
    with `cudaMemcpyToSymbol`. (Hand-rolling: you'd otherwise stream the weights from
    global memory per thread — far more bandwidth.)
  - **Global** — the two `int8` window arrays (`ref`, `alt`). Each variant's `L=21` bytes
    are contiguous. This kernel is **compute-bound per thread** (~1360 double FMAs), so
    the small global reads are easily hidden by occupancy.
  - **Registers** — the per-thread `pooled[K]` accumulator and the running conv sums live
    in registers; the activation map is never materialised (we fuse conv→ReLU→pool).
  - **No shared memory, no atomics** — outputs are fully independent, the cleanest mapping.

> **Why this scales.** Real atlases have tens of millions of variants. One thread per
> variant means the GPU's thousands of cores chew through the batch with perfect
> parallelism; doubling the card's cores ~doubles throughput. Production replaces our hand
> CNN with cuDNN/TensorRT kernels on Tensor Cores, but the *parallel-over-variants*
> decomposition is identical.

---

## 5. Numerical considerations

- **Precision: FP64 throughout.** Both the conv accumulation and the sigmoid run in
  `double`. We do not need FP64 for accuracy here — we use it so the CPU and GPU match
  tightly enough to make verification a *strict* test (see §6). A production model would
  use BF16/FP16 on Tensor Cores and accept a looser tolerance.
- **Stable sigmoid.** `vep_sigmoid` branches on the sign of `x`: for `x ≥ 0` it computes
  `1/(1+e^{−x})` (exponent ≤ 0, no overflow); for `x < 0` it computes `e^{x}/(1+e^{x})`.
  The naive single-branch `1/(1+e^{−x})` overflows `exp` for large negative `x` and would
  make the CPU and GPU disagree near the extremes. The branch is data-dependent but
  **deterministic** and identical on both sides.
- **Determinism / no atomics.** Each thread writes exactly one `effect[i]`; there is no
  cross-thread reduction, so there is **no floating-point summation-order nondeterminism**
  (contrast `5.01`/`11.09`, which must accumulate in integers to stay reproducible). The
  only CPU↔GPU difference is the GPU's fused-multiply-add contracting `a·b+c` to one
  rounding vs. the host's two — a ~1 ULP effect.
- **Ranking on the trusted side.** `main.cu` ranks on the **CPU** scores, so the printed
  order is bit-stable across GPUs even though the GPU deltas differ in the last bit. Ties
  break by lower index for full determinism.
- **stdout vs. stderr.** Deterministic results (the ranking, the PASS verdict) go to
  **stdout** (diffed by the demo). Timings and the measured error go to **stderr** (shown,
  not diffed) — the determinism rule of PATTERNS.md §3.

---

## 6. How we verify correctness

Two independent checks:

1. **CPU ↔ GPU agreement.** `main.cu` computes `max_i |effect_cpu[i] − effect_gpu[i]|`
   and asserts it is `≤ 1e-9`. Because both sides call the **same** `vep_model.h` core,
   the only divergence is FMA rounding; the observed error is `~2.8e-17` (machine epsilon,
   ~0.1 ULP), thousands of times under the bound. This is the "machine-precision class"
   tolerance of PATTERNS.md §4 — honest, not hand-wavy.
2. **A known planted answer.** The synthetic sample (`scripts/make_synthetic.py`) embeds
   variants whose alternate allele *completes* the deleterious motif `CAGCT` at the centre
   of the window. Since the model's filter 0 is tuned to that 5-mer with a strong positive
   dense weight, those variants **must** top the ranking — and they do (`#1`–`#3` in
   `expected_output.txt`). The protective-motif variants correctly sink. This validates the
   *science of the toy model*, not just CPU==GPU. (PATTERNS.md §6.)

Edge cases handled: an all-non-firing filter pools to `0` (ReLU floor); a degenerate
window is rejected by the loader (`load_variants` validates width, bases, and that the
window centre equals the stated reference allele).

---

## 7. Where this sits in the real world

| Aspect | This teaching project | Production (AlphaMissense / Enformer / ESM-1v) |
|---|---|---|
| Model | fixed 8-filter 1-D CNN, ~few KB | trained CNN+attention / transformer, 10⁶–10⁹ params |
| Weights | hand-designed, synthetic | learned from ClinVar/gnomAD/DMS/MSA at scale |
| Inference | hand-rolled FP64, one thread/variant | cuDNN / TensorRT kernels, Tensor-Core BF16/FP16 |
| Context | 21 bases | 200 kb (Enformer); full protein (ESM) |
| Effect | `σ`-prob difference | calibrated pathogenicity / log-odds ratio |
| Throughput trick | constant-memory weights, grid-stride | CUDA Graphs for low-latency repeated launches; batching |

**What you would change to grow this up:**
- Swap the hand CNN for a real architecture and load **trained** weights; call **cuDNN**
  for the conv/dense layers (it picks tuned algorithms; see PATTERNS.md §5 on using a
  library without it being a black box) and **TensorRT** to fuse + quantise for deploy.
- Move to **Tensor Cores** (BF16) for the matmuls — an order-of-magnitude throughput win
  on Ampere/Ada, at the cost of a looser numerical tolerance.
- Use **CUDA Graphs** to amortise launch overhead across the millions of tiny repeated
  inferences (the "low-latency repeated inference" the catalog names).
- Calibrate the output against ClinVar labels so the score is an actual probability — and
  validate it like a clinical model, which is **far** beyond an educational repo.

Even after all that, the decomposition you learned here — *variants are independent, score
each as a ref/alt forward-pass pair, take the difference* — is exactly what the big systems
do. This project is that idea, small enough to read.
