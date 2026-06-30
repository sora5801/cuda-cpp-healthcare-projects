# THEORY — 2.1 Protein Structure Prediction Inference (AlphaFold-class)

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> **Reduced-scope teaching version.** We implement and verify the single most
> important GPU operation inside AlphaFold/ESMFold — **scaled dot-product
> self-attention** — and describe the rest of the pipeline in §7.
>
> _Educational only — not for clinical use._

---

## 1. The science

A protein is a linear chain of amino-acid **residues** that folds into a precise
3-D shape. That shape determines what the protein *does* — which molecules it
binds, what reactions it catalyzes, how a drug might inhibit it. Determining the
shape experimentally (X-ray crystallography, cryo-EM) is slow and expensive, so
for decades "predict the 3-D structure from the 1-D sequence" was biology's
grand-challenge problem.

**AlphaFold2** (2021) effectively solved it for single chains, reaching
experimental accuracy on many targets. Its successors — **OpenFold** (an open
PyTorch reimplementation), **ESMFold** (which replaces the evolutionary input with
a protein *language model*), **RoseTTAFold**, and **AlphaFold3 / Boltz-1** (which
add ligands, nucleic acids, and diffusion-based generation) — all share one engine:
a deep stack of **transformer attention** layers.

The key biological intuition behind attention here: residues that are far apart in
the 1-D sequence are often close in 3-D, and their amino acids **co-evolve** (a
mutation in one is compensated by a mutation in its 3-D neighbor). Attention is the
mechanism that lets the network discover and use these long-range relationships —
every residue can directly query every other residue, regardless of sequence
distance. This project implements exactly that query-every-other-residue step.

## 2. The math

We compute one head of **scaled dot-product self-attention** (Vaswani et al.,
*Attention Is All You Need*, 2017), the operation AlphaFold's Evoformer applies
over and over.

**Inputs.** A residue representation for a length-`L` protein, already projected
into three matrices (each row is one residue's length-`d` feature vector):

- `Q ∈ ℝ^{L×d}` — **queries**: "what is residue `i` looking for?"
- `K ∈ ℝ^{L×d}` — **keys**: "what does residue `j` offer?"
- `V ∈ ℝ^{L×d}` — **values**: "what does residue `j` contribute if attended to?"

Symbols: `L` = sequence length (residues, `L=6` in the sample); `d` = feature
width per residue (`d = D_MODEL = 32`, dimensionless); indices `i, j ∈ [0, L)`
over residues, `c ∈ [0, d)` over channels.

**Output.** `Out ∈ ℝ^{L×d}`, the context-mixed representation:

```
S[i][j]   = (Q[i] · K[j]) / sqrt(d)              # scaled affinity, a scalar
w[i][j]   = exp(S[i][j]) / Σ_j' exp(S[i][j'])    # softmax over j (Σ_j w[i][j] = 1)
Out[i][c] = Σ_j  w[i][j] · V[j][c]               # weighted average of values
```

The `1 / sqrt(d)` factor is the **"scaled"** in scaled dot-product attention:
without it, a dot product of two `d`-vectors grows in magnitude ~`√d`, pushing
softmax toward a one-hot spike (over-confident, vanishing gradients in training).
Dividing by `sqrt(d)` keeps the score variance ~constant in `d`.

**Objective.** There is no optimization here — it is a *forward* (inference)
computation. Given Q, K, V, produce `Out`. In a full model this is one of dozens
of such layers; the layer *weights* (the projections that make Q, K, V) are what
training learns.

## 3. The algorithm

For each query residue `i` (rows are independent of one another):

```
1. for j in 0..L-1:  S[j] = scaled_score(Q[i], K[j])      # O(L·d) dot products
2. m = max_j S[j]                                          # O(L)   for stability
3. for j: e[j] = exp(S[j] - m);  denom += e[j]             # O(L)
4. for c in 0..d-1:  Out[i][c] = Σ_j (e[j]/denom)·V[j][c]  # O(L·d)
```

**Complexity.**

| Quantity | Cost |
|---|---|
| Scores for one row | `O(L·d)` multiply-adds |
| Softmax for one row | `O(L)` |
| Value mix for one row | `O(L·d)` |
| **All `L` rows (serial)** | **`O(L² · d)` time**, `O(L)` extra space per row |
| Parallel **work** | `O(L² · d)` (same total) |
| Parallel **depth** | `O(log L + L)` per row: a tree reduction for max/sum, then a length-`L` value accumulation |

The `O(L²)` scaling in sequence length is the crux: doubling the protein
quadruples the attention cost, and real models stack this dozens of times over
MSAs of hundreds of sequences. That is the wall that makes GPUs mandatory (and
that FlashAttention attacks; see §7).

**Arithmetic intensity / data access.** The scores re-read all of `K` for every
query row, and the value mix re-reads all of `V` — the schoolbook kernel is
**memory-bandwidth-bound** for large `L`. Production kernels tile Q/K/V through
fast on-chip memory to raise arithmetic intensity.

## 4. The GPU mapping

**Decomposition: one thread *block* per output row.** The `L` rows of `Out` are
independent, so block `b` owns query residue `i = b`. Inside the block, the
`blockDim.x` threads cooperate on that one row's three phases.

```
grid  = L blocks                         (one per query residue / output row)
block = D_MODEL = 32 threads             (one warp; also one thread per output channel)

         block b  ==  residue i = b
   ┌───────────────────────────────────────────────┐
   │ phase 1  threads stride over j: S[j]=Q[i]·K[j] │  -> shared mem  s_w[0..L-1]
   │          block_reduce_max  -> row_max          │  (tree reduction in s_red)
   │ phase 2  s_w[j] = exp(S[j]-row_max); Σ -> denom│  -> shared mem
   │          block_reduce_sum  -> denom            │
   │ phase 3  thread c sums Σ_j (s_w[j]/denom)·V[j][c]  -> Out[i][c]
   └───────────────────────────────────────────────┘
```

**Thread-to-data mapping.**
- *Phase 1 & 2* use a **grid-stride over `j`**: thread `t` handles residues
  `j = t, t+blockDim.x, …`, so any `L` is covered by 32 threads.
- *Phase 3* maps **thread `t` → output channel `c = t`** (valid because
  `blockDim.x == D_MODEL`). Each thread walks `j = 0..L-1` **in increasing order**,
  accumulating `Σ_j w[j]·V[j][c]` — the *same order* the CPU uses, which is what
  makes the two results match to rounding (§5, §6).

**Why this block size.** `D_MODEL = 32` equals one warp: in phase 3 every lane is
an active output channel (no divergence, no idle lanes), and `threadIdx.x` doubles
as the channel index. For a larger `d` you would round the block up to a warp
multiple and have each thread own several channels.

**Memory hierarchy.**
- **Global memory:** Q, K, V, Out (the bulk data). Reads of consecutive `K[j]`
  rows by consecutive `j` are reasonably coalesced.
- **Shared memory (dynamic):** the per-row scores `s_w[0..L-1]` (reused as softmax
  weights) plus `blockDim.x` doubles of reduction scratch. Shared memory is the
  right home because all 32 threads in the block read/write these same `L` values
  many times — keeping them on-chip avoids `O(L)` global round-trips per phase.
- **Registers:** each phase-3 thread accumulates its channel in a register.
- **No constant/texture** memory is needed for this small kernel.

**Occupancy & the shared-memory limit.** Shared memory used is
`(L + blockDim.x) · sizeof(double)` bytes. For the sample (`L=6`) that is ~336 B —
trivial. But it grows linearly in `L`, so a naive single-block-per-row kernel
would exceed the ~48 KB default shared budget around `L ≈ 6000` residues. That
limit is *exactly* what FlashAttention removes by tiling (§7) — a real lesson the
small kernel makes concrete.

**CUDA libraries / no black boxes.** This project deliberately uses **no** library
kernel: the dot products, the reduction, and the softmax are all hand-written so
every FLOP is visible. A production path would instead express the scores as a
batched GEMM (`Q·Kᵀ`) via **cuBLAS** and the value mix as another GEMM
(`w·V`) — two dense matrix multiplies — and use **cuDNN**'s fused multi-head
attention or a **FlashAttention** kernel for the whole thing. Writing those by
hand efficiently means tiling into shared memory and managing the online softmax;
we do the un-tiled version so the math is legible first.

```
ASCII: phase-3 channel ownership (blockDim.x == d == 32)

 thread t:   0    1    2   ...   31
 owns col:  c=0  c=1  c=2  ...  c=31     of Out[i][·]
 each loops j = 0,1,...,L-1  accumulating  w[j]*V[j][t]
```

## 5. Numerical considerations

- **Precision.** Inputs/outputs are FP32 (as real inference largely is), but every
  *accumulation* — the dot products, the exponent sum, the value mix — is done in
  **`double`** inside `attention_core.h`. Using double for the reductions removes
  most of the rounding noise and, crucially, lets the CPU and GPU agree closely.
- **Softmax stability.** Raw `exp(S)` overflows to `+inf` for modest `S`, giving
  `inf/inf = NaN`. We subtract the row maximum first: `exp(S - max) ∈ (0,1]`, which
  is the mathematically identical softmax with no overflow. CPU and GPU subtract
  the **same** `max`, so the shifted exponentials match.
- **Race conditions / atomics.** **None.** Each block writes a disjoint output row
  and each thread a disjoint channel, so there are zero write conflicts — no
  `atomicAdd` anywhere. The only cross-thread communication is the two
  `__syncthreads()`-guarded shared-memory tree reductions.
- **Determinism.** The dangerous source of nondeterminism in GPU reductions is
  *floating-point summation order* (float adds are not associative). We sidestep it
  two ways: (a) the reductions accumulate in `double`, and (b) the value-mix sum in
  phase 3 walks `j` in the **same increasing order** the CPU uses. Result: `stdout`
  is byte-identical every run, and matches the CPU to ~`1e-7`.

## 6. How we verify correctness

The CPU reference [`reference_cpu.cpp`](src/reference_cpu.cpp) computes the *same*
three-line attention definition serially, calling the *same* `attention_core.h`
primitives the kernel calls. [`main.cu`](src/main.cu) runs both and reports
`max_abs_err = max_{i,c} |Out_cpu[i][c] − Out_gpu[i][c]|`.

- **Tolerance: `1e-5`.** This is an *honest FP32* bound, not bit-exactness. Even
  though we accumulate in double, the final `Out` is stored as `float`, and the
  GPU and host compilers may contract/round the surrounding casts slightly
  differently. The observed error on the sample is `~4.8e-7`, comfortably inside
  `1e-5` (see `docs/PATTERNS.md §4` for why long FP32 pipelines warrant a small
  physical tolerance rather than `== 0`).
- **A stronger, *semantic* check.** The synthetic data is built so each residue's
  `Q`/`K` peak in a unique channel, which means **every residue must attend most
  to itself**. The demo prints that argmax per row, and it is residue `i → i` for
  all `i` — validating not just "CPU == GPU" but "the attention mechanism does the
  right thing" (`docs/PATTERNS.md §6`).
- **Edge cases.** The loader rejects a wrong feature width (`d ≠ D_MODEL`), a
  non-positive `L`, and truncated data; the softmax guards the all-equal-scores
  case (uniform weights) naturally. Why is agreement convincing? Because the CPU
  path is short, serial, and obviously a transcription of the textbook formula —
  an independent implementation reaching the same numbers is strong evidence the
  parallel one is right.

## 7. Where this sits in the real world

This kernel is **one attention head**. A real AlphaFold-class system wraps it in a
great deal more (all described here, none implemented):

- **Learned projections & multi-head attention.** Q, K, V come from learned linear
  layers; attention runs in parallel across `h` heads (different learned
  subspaces) whose outputs are concatenated and re-projected.
- **The Evoformer (AF2).** Two coupled representations — an **MSA** matrix
  (sequences × residues) and a **pair** matrix (residue × residue) — are refined by
  48 blocks of: **row** attention, **column** attention, **triangle multiplicative
  updates**, and **triangle attention**, which enforce geometric consistency
  (if `i–j` and `j–k` distances are known, `i–k` is constrained).
- **The Structure Module.** **Invariant Point Attention (IPA)** — attention that is
  equivariant to 3-D rotations/translations — turns the abstract representation
  into actual atomic coordinates (residue frames + side-chain torsions).
- **Recycling.** The whole stack is run ~3–4 times, feeding its output back as
  input, which sharpens the prediction.
- **Confidence heads.** **pLDDT** (per-residue confidence) and **PAE** (pairwise
  alignment error) are predicted alongside the structure.
- **ESMFold** drops the MSA entirely, replacing the evolutionary signal with a
  ~15-billion-parameter protein **language model**, trading some accuracy for a
  10–60× speedup. **AlphaFold3 / Boltz-1** replace the structure module with a
  **diffusion** generator and handle ligands and complexes.
- **The kernel they actually run.** Production attention is **FlashAttention**: the
  identical math, but tiled so the `L × L` score matrix is never written to global
  memory. It keeps a running max and running denominator (the *online softmax*) as
  it streams K/V tiles, fitting attention for thousands of residues into on-chip
  memory. Exercise 3 in the README walks you toward it; our §4 shared-memory limit
  is precisely the problem it solves.

---

## References

- Vaswani et al., *Attention Is All You Need*, NeurIPS 2017 — the scaled
  dot-product attention this project implements.
- Jumper et al., *Highly accurate protein structure prediction with AlphaFold*,
  Nature 2021 — the Evoformer, IPA, recycling, and confidence heads (§7).
- Lin et al., *Evolutionary-scale prediction … with a language model* (ESMFold),
  Science 2023 — MSA-free attention-only prediction.
- Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention*, 2022 —
  the tiled/online-softmax kernel real systems run.
- Code to study (do not copy wholesale): AlphaFold
  (<https://github.com/google-deepmind/alphafold>), OpenFold
  (<https://github.com/aqlaboratory/openfold>), ESM
  (<https://github.com/facebookresearch/esm>), Boltz-1
  (<https://github.com/jwohlwend/boltz>).
