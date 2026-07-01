# THEORY — 4.21 MR Fingerprinting Reconstruction

> Deep didactic companion to the code. Read after the project `README.md`, then
> walk the code in the order `main.cu → kernels.cuh → kernels.cu →
> reference_cpu.cpp`, with `mrf_core.h` open for the shared per-element math.

---

## The science

**Conventional MRI** produces *qualitative* images: contrast depends on the
scanner, the sequence, and the operator, so a "bright" pixel has no absolute
meaning. **Quantitative MRI** instead measures physical tissue constants —
chiefly the two relaxation times:

- **T1** (longitudinal / spin–lattice relaxation, ms): how fast the magnetization
  realigns with the main field after a pulse. Fat is short-T1, cerebrospinal
  fluid (CSF) is very long-T1.
- **T2** (transverse / spin–spin relaxation, ms): how fast the in-plane signal
  dephases. CSF is long-T2, white matter is short-T2.

**Magnetic Resonance Fingerprinting (MRF)** (Ma et al., *Nature* 2013) measures
T1 and T2 *simultaneously and fast*. The trick: instead of a periodic sequence
that reaches a boring steady state, MRF plays a **pseudorandom** train of flip
angles and repetition times. Under this deliberately varied excitation, every
tissue emits a **distinctive signal evolution over time — its "fingerprint"**.
Two tissues that look identical in one static image trace out clearly different
curves across the ~1000-frame train.

Reconstruction is then **pattern recognition**: precompute a *dictionary* of
fingerprints (one per candidate `(T1, T2)`), and for each image voxel find the
dictionary entry whose fingerprint best matches the measured signal. The matched
entry's `(T1, T2)` are the voxel's quantitative values. This project implements
that **dictionary-matching** step — the compute bottleneck — on the GPU.

> **Educational only.** The model here is a deliberately simplified teaching
> version (see *Where this sits in the real world*). It is not for diagnosis.

---

## The math

### One tissue's fingerprint (the forward model)

A tissue's magnetization is a 3-vector; MRF's full physics is the **Bloch
equations** (or the discrete **Extended Phase Graph**, EPG). For teaching we use
a compact real-valued recursion that keeps the three effects a learner must
understand — RF tipping, T2 decay, T1 recovery — and drops the rest.

Let `Mz` be the longitudinal magnetization (equilibrium `M0 = 1`). The scan
starts from an **inversion**, `Mz = −1` (a 180° preparation pulse; this is what
makes early frames strongly T1-weighted). Frame `t` has flip angle `α_t`,
repetition time `TR_t`, echo time `TE_t`. One frame does:

1. **Tip + read out.** The pulse rotates `Mz` into the transverse plane; the
   sampled signal is the transverse component after decaying for `TE_t`:

   $$ s_t = M_z \sin(\alpha_t)\, e^{-TE_t / T_2}. $$

2. **Recover.** The untipped longitudinal part `M_z\cos(α_t)` relaxes toward `M0`
   over the rest of `TR_t`:

   $$ M_z \leftarrow 1 - \big(1 - M_z\cos(\alpha_t)\big)\, e^{-TR_t / T_1}. $$

Iterating this for `t = 0 … T−1` yields the length-`T` fingerprint
`f(T1, T2) = (s_0, …, s_{T−1})`. This is `mrf::bloch_step` /
`mrf::simulate_atom` in [`src/mrf_core.h`](src/mrf_core.h), shared verbatim by
CPU and GPU. **Where the two relaxation times enter:** `T1` shapes the recovery
term (dominant in the early inversion transient); `T2` shapes the per-frame decay
`e^{-TE_t/T_2}` (which is why a *widely varying* `TE_t` is needed to make `T2`
identifiable — see the synthetic schedule).

### Matching (the inverse problem)

A measured voxel signal is `y = c · f(T1*, T2*) + noise`, where `c > 0` is an
unknown scalar (proton density × receive gain). We do not know `c` and do not
need it to identify the tissue — only the **shape** (direction) of `y` matters.
So we **L2-normalize** both the voxel signal and every atom to unit length, and
the match score becomes a **cosine**:

$$ \text{score}(y, d) = \frac{\langle y,\, f_d\rangle}{\lVert y\rVert\,\lVert f_d\rVert}
   = \langle \hat y,\, \hat f_d\rangle \in [-1, 1]. $$

The reconstruction picks, per voxel, the atom of **maximum cosine**:

$$ d^*(y) = \arg\max_{d}\ \langle \hat y,\, \hat f_d\rangle. $$

The matched `(T1_{d^*}, T2_{d^*})` are the parameter estimates, and the recovered
scale `c ≈ \lVert y\rVert · \text{score}` is a bonus **proton-density** map.

### The bottleneck as a matrix product

Stack the `V` normalized voxel signals as rows of `Ŷ ∈ ℝ^{V×T}` and the `D`
normalized atoms as rows of `F̂ ∈ ℝ^{D×T}`. Then **all** `V·D` cosines are the
entries of one matrix product:

$$ S = \hat Y\, \hat F^{\mathsf T} \in \mathbb{R}^{V\times D}, \qquad
   S_{v,d} = \langle \hat y_v, \hat f_d\rangle. $$

Per-voxel `argmax` over the rows of `S` finishes the job. At clinical scale
(`V ~ 10⁵`, `D ~ 10⁵`, `T ~ 10³`) this is ~**10¹¹ inner products** — precisely the
kind of dense linear algebra a GPU GEMM devours.

---

## The algorithm (and complexity)

```
build dictionary:   for each atom d:  f_d = simulate(T1_d, T2_d);  normalize(f_d)
normalize signals:  for each voxel v: scale_v = ||y_v||;  ŷ_v = y_v / scale_v
match:              S = Ŷ · F̂ᵀ                      # V×D cosine matrix (GEMM)
                    for each voxel v: d*(v) = argmax_d S[v,d]
```

| Step | Serial cost | Parallel structure |
|------|-------------|--------------------|
| Build dictionary | `O(D·T)` | `D` independent simulations (thread per atom) |
| Normalize signals | `O(V·T)` | `V` independent normalizations (thread per voxel) |
| **Match (GEMM)** | **`O(V·D·T)`** | dense matrix multiply → **cuBLAS SGEMM** |
| Argmax | `O(V·D)` | `V` independent row-reductions (thread per voxel) |

The **match dominates** (`V·D·T ≫ D·T, V·T`); it is the step the GPU exists for.
The CPU reference ([`reference_cpu.cpp`](src/reference_cpu.cpp)) writes the match
as an obvious triple loop; the GPU replaces that loop with a single library call.

---

## GPU mapping

The pipeline is four device stages, orchestrated by `gpu_reconstruct` in
[`src/kernels.cu`](src/kernels.cu):

**Stage 1 — `build_dict_kernel` (independent jobs, PATTERNS.md §1).**
One thread per atom runs the shared `mrf::simulate_atom` and normalizes in place.
Grid = `ceil(D/256)`, block = 256. Each atom occupies a contiguous `T`-float row
of `dict_norm` (row-major `D×T`).

**Stage 2 — `norm_sig_kernel`.** One thread per voxel L2-normalizes its signal
row and stores the norm (the proton-density scale).

**Stage 3a — cuBLAS SGEMM (the headline; PATTERNS.md §5).**
`S = Ŷ · F̂ᵀ`. Both `Ŷ` (`V×T`) and `F̂` (`D×T`) are **row-major**, but cuBLAS is
**column-major**. The zero-copy trick: a row-major `V×T` buffer is bit-identical
to a column-major `T×V` matrix. Viewing `Ŷ` as column-major `SigC` (T×V) and `F̂`
as column-major `DictC` (T×D), the product we want is

$$ S_c = \text{SigC}^{\mathsf T}\cdot \text{DictC}, $$

i.e. `cublasSgemm(op(A)=T, op(B)=N, m=V, n=D, k=T, A=SigC lda=T, B=DictC ldb=T,
C=S ldc=V)`. We keep `S` **column-major** (`V` rows, `D` cols); the argmax kernel
reads `S[d*V + v]`. Writing a competitive GEMM by hand means shared-memory
tiling, register blocking, and bank-conflict-free loads — cuBLAS already does all
of that, which is why we use it and explain it rather than hide it.

**Stage 3b — `argmax_kernel`.** One thread per voxel scans its `D` scores
(column-strided reads, stride `V`) and keeps the maximum, breaking ties by
**smallest atom index** (strict `>`), matching the CPU exactly.

**Memory hierarchy used.** Global memory holds the dictionary, signals, and score
matrix. The schedule (`α, TR, TE`) is read by every thread of Stage 1 and is a
natural **constant-memory** candidate (broadcast cache); we leave it in global
for teaching simplicity and flag the optimization inline. Registers hold the
per-thread accumulators. Shared memory is used *inside* cuBLAS's tiled GEMM (not
in our own kernels) — a good thing to inspect in Nsight.

---

## Numerical considerations

- **Precision.** The forward model runs in **double** (relaxation exponentials
  need the headroom), then stores fingerprints as **float** for the SGEMM (MRF
  matching is single-precision in practice; the dictionary is the memory-limited
  object). Inner products accumulate in `double` inside `mrf::dot` on the CPU.
- **Why the dictionary matches CPU==GPU to 0.** Both sides feed the *same* double
  `(T1, T2)` through the *same* `mrf::simulate_atom` and `mrf::normalize_inplace`
  and cast to float identically, so the normalized dictionary is **bit-identical**
  (observed worst diff `0.0e0`).
- **Why the cosines differ slightly.** cuBLAS SGEMM sums each length-`T` inner
  product in a different **order** than the CPU's serial loop and uses fused
  multiply-add. Float addition is not associative, so the cosines agree only to
  ~`1e-7`, not bit-exactly — a real, teachable effect (PATTERNS.md §4).
- **Determinism of the *result*.** The reported answer is driven by the `argmax`
  **integer index**, not the float score. A float wobble of `1e-7` could flip an
  argmax *only near a tie*. The synthetic dictionary is **greedily pruned** so no
  two atoms are within cosine `0.995`, and the demo's smallest top-1-vs-top-2
  margin is ~`4e-3` — **~10⁴× larger** than the float error. So every voxel's
  index is decided far away from any tie, the GPU and CPU agree on **all** indices
  (0 mismatches), and stdout is byte-for-byte reproducible.
- **Atomics.** None needed: no cross-thread reductions in our kernels, so there is
  no float-atomic nondeterminism to worry about.

---

## How we verify correctness

The demo runs the CPU reference and the GPU pipeline on the same input and checks
three things (`main.cu`), with documented tolerances (PATTERNS.md §4):

1. **Dictionary** — worst entrywise `|dict_gpu − dict_cpu|` ≤ `1e-4` (observed
   `0.0e0`, because the shared double math casts to float identically).
2. **Best-atom index** — must match the CPU **exactly** for **every** voxel
   (integer equality). This is the strong check; it holds because of the margin
   argument above.
3. **Cosine score** — worst `|cos_gpu − cos_cpu|` ≤ `1e-4` (observed ~`2e-7`).

A second, *scientific* check beyond CPU==GPU: because each synthetic voxel was
drawn from a **known** atom, we report **reconstruction accuracy** against ground
truth (64/64 recovered) and the **median T1/T2 error** (0.0 ms) — validating that
the method recovers the right physics, not merely that two implementations agree.
There is also a spot-check that the SGEMM's voxel-0 score row equals a hand-rolled
CPU inner product for every atom.

---

## Where this sits in the real world

This is a **reduced-scope teaching version**. Production MRF differs in ways worth
knowing:

- **Forward model.** Real dictionaries come from a full **Bloch** or **Extended
  Phase Graph (EPG)** simulation that tracks the complex transverse/longitudinal
  states, models RF spoiling, slice profile, `B1+` inhomogeneity, off-resonance
  (`B0`), and diffusion. Our closed-form real recursion captures the qualitative
  T1/T2 sensitivity only. Differentiable simulators (e.g. **MRzero**) even
  optimize the *sequence* for parameter separability via the Cramér–Rao bound.
- **Dictionary size.** Real dictionaries hold `10⁵–10⁶` atoms over a dense
  `(T1, T2[, B0, B1])` grid; ours has 22 well-separated atoms. Bigger dictionaries
  reintroduce near-collinear atoms and make matching genuinely ill-conditioned —
  the reason for **low-rank / SVD compression** of the temporal dimension.
- **Non-Cartesian acquisition.** MRF usually uses spiral k-space, so
  reconstruction needs a **NUFFT** (a **cuFFT** step) per frame before matching,
  often folded into an iterative **low-rank subspace** or **ADMM** reconstruction
  rather than a single dictionary pass.
- **The matcher itself, though, is exactly this GEMM.** The `S = Ŷ · F̂ᵀ`
  dictionary match — computed as one big cuBLAS SGEMM (or a batched GEMM across
  slices) — is the real, production compute pattern the catalog names, and it is
  what this project teaches faithfully. Tools that do it at scale: **BART**
  (low-rank subspace MRF), **SigPy** (NUFFT-based MRF), and various PyTorch
  dictionary-matching implementations.

## Further reading

- Ma, Gulani, Seiberlich, et al. "Magnetic resonance fingerprinting." *Nature*
  495 (2013): 187–192 — the founding paper.
- **BART** — <https://github.com/mrirecon/bart> (low-rank subspace MRF recon).
- **SigPy** — <https://github.com/mikgroup/sigpy> (NUFFT-based MRF recon).
- **MRzero** — differentiable MR sequence simulation for MRF sequence design.
