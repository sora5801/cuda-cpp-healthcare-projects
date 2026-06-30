# THEORY — 3.18 Protein Language Model Inference

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

A protein is a chain of amino-acid **residues**. Its 1-D sequence (a string over
the 20-letter amino-acid alphabet `ACDEFGHIKLMNPQRSTVWY`) folds into a 3-D
structure that determines its function. The central problem of computational
biology is to read meaning out of that sequence: which residues are in contact,
what the fold is, how a mutation changes function.

For decades the dominant signal came from **evolution**: by aligning a protein
against thousands of homologs (a multiple sequence alignment, MSA), co-varying
columns reveal residues that touch in 3-D. AlphaFold2 built its accuracy on MSAs.

A **protein language model (PLM)** takes a different route. Borrowing the
Transformer architecture from natural-language processing, models like Meta's
**ESM-2** are trained on hundreds of millions of raw sequences with a *masked
language modeling* objective: hide some residues, predict them from context.
To do that well the model must internalize the same evolutionary and biophysical
constraints — so its learned per-residue **embeddings** encode structure,
function, and the effect of mutations, *from a single sequence with no MSA*.
**ESMFold** stacks ESM-2 as a trunk and predicts 3-D structure directly, far
faster than MSA-based methods.

The beating heart of every Transformer layer is **self-attention**: a mechanism
that lets each residue gather information from every other residue, weighted by
how relevant they are. That is the one computation this project implements and
maps to the GPU. Everything else in a PLM (residual connections, normalization,
the feed-forward MLP, the structure module) is scaffolding around attention.

## 2. The math

We implement one **multi-head self-attention** block. Let:

- `L` = sequence length (number of residues). Here `L = 24`.
- `d` = `d_model`, the embedding width. Here `d = 32`.
- `H` = number of heads; `d_head = d/H` the per-head width. Here `H = 4`, `d_head = 8`.

**Input.** Each residue `i` has an embedding row `X[i] ∈ ℝ^d`; stacked, `X ∈ ℝ^{L×d}`.
(In a real PLM `X` is a learned lookup + positional encoding; here `X[i,f] =
embed_value(token_i, f)`, a deterministic hash — §5.)

**Projections.** Three learned weight matrices `Wq, Wk, Wv ∈ ℝ^{d×d}` produce

```
Q = X·Wq ,   K = X·Wk ,   V = X·Wv          (each ℝ^{L×d})
```

The columns are partitioned into `H` heads: head `h` uses columns
`[h·d_head, (h+1)·d_head)` of `Q`, `K`, `V`, written `Q_h, K_h, V_h ∈ ℝ^{L×d_head}`.

**Scaled dot-product attention (per head).** For head `h`:

```
S_h = Q_h · K_hᵀ / √d_head                  (ℝ^{L×L} logits)
A_h = softmax(S_h, over the key axis)        (ℝ^{L×L}, each row sums to 1)
O_h = A_h · V_h                              (ℝ^{L×d_head} head output)
```

`S_h[i,j] = (q_i · k_j)/√d_head` is how strongly query residue `i` matches key
residue `j`. The `1/√d_head` factor keeps the dot product's variance ≈ 1 so the
softmax does not saturate (Vaswani et al. 2017). The softmax over `j` turns the
logit row into a probability distribution — residue `i`'s **attention** over all
residues. `O_h[i] = Σ_j A_h[i,j]·V_h[j]` is then a weighted blend of every
residue's value vector.

**Combine heads.** Concatenate the head outputs back to width `d`:
`Z = [O_0 | O_1 | … | O_{H-1}] ∈ ℝ^{L×d}`, then apply the output projection
`Wo ∈ ℝ^{d×d}`:

```
Y = Z · Wo                                   (ℝ^{L×d} block output)
```

**What we report.** The per-residue output-embedding norm `‖Y[i]‖₂`, and for head
0 the **argmax over keys** of `A_0[i,·]` — the residue `i` attends to most, a
compact "contact-like" readout.

The numerically-stable softmax we actually compute (to avoid `exp` overflow) is

```
m   = max_j S[i,j]
A[i,j] = exp(S[i,j] − m) / Σ_l exp(S[i,l] − m)
```

which is mathematically identical to the naive form but never exponentiates a
positive number.

## 3. The algorithm

Pseudocode for one block (serial):

```
for i in 0..L:                       # each residue
  q_i = project(X[i], Wq)            # O(d²) over the whole d; per-head slice used below
for each head h:
  for i in 0..L:                     # each query
    for j in 0..L:  S[j] = (q_i·k_j)/√d_head     # O(L·d_head)
    A = softmax(S)                                 # O(L)
    for t in 0..d_head:                            # value blend
      O_h[i,t] = Σ_j A[j]·V_h[j,t]                 # O(L·d_head)
Y = Z·Wo                                           # O(L·d²)
```

**Complexity.** The projections are `O(L·d²)`. The attention scores and blends are
`O(H · L² · d_head) = O(L²·d)`. So the block is `O(L·d² + L²·d)`. For long proteins
the `L²·d` attention term dominates — this is the famous **quadratic-in-length**
cost of attention, and the reason long-context Transformers are expensive.

**Arithmetic intensity / access pattern.** Every step is a dense matrix product or
a row reduction (softmax). These are high-arithmetic-intensity GEMMs: lots of
multiply-adds per byte loaded, which is exactly what GPUs (and their Tensor Cores)
are built for. The data-access pattern is regular and contiguous (row-major
matrices), so memory coalescing is natural.

## 4. The algorithm → GPU mapping

We keep each **attention row independent** and assign **one thread-block per
`(head, query-row)`**. The grid is 2-D: `grid.x = L` query residues,
`grid.y = H` heads. Inside a block, `ATTN_THREADS = 128` threads cooperate over
the `L` keys.

```
        grid.y = heads (H)
        ┌───────────────────────────────┐
grid.x  │ block(0,0) block(0,1) ...      │   block(i,h) computes
 = L    │ block(1,0) block(1,1) ...      │   attention ROW i for head h
queries │   ...                          │
        └───────────────────────────────┘

   inside block(i,h):  threads t=0..127 stride over keys j
   ┌── Phase 1: logits ─────────────────────────────────────┐
   │ s_logit[j] = q_i · k_j / √d_head     (shared memory)    │
   ├── Phase 2: stable softmax (block reductions) ──────────┤
   │ row_max = max_j s_logit[j]   (tree max in s_red)        │
   │ s_logit[j] = exp(s_logit[j] − row_max)                  │
   │ inv = 1 / Σ_j s_logit[j]     (tree sum in s_red)        │
   │ s_logit[j] *= inv            (now = A[i,j])             │
   ├── Phase 3: value blend ────────────────────────────────┤
   │ thread t<d_head:  Z[i, off+t] = Σ_j s_logit[j]·V_h[j,t] │
   └────────────────────────────────────────────────────────┘
```

**Thread-to-data mapping.**
- Phase 1: thread `t` owns keys `j = t, t+128, …`; it builds `q_i` and `k_j`'s
  `d_head` slices via `proj_one()` and writes the scaled logit to `s_logit[j]`.
- Phase 2: a classic two-pass block reduction — first a tree **max** over the
  partials, then a tree **sum** of the exponentials — both in shared memory.
- Phase 3: thread `t` (for `t < d_head`) owns output dimension `t` and sums the
  value blend over all `L` keys sequentially.

**Launch configuration.** `ATTN_THREADS = 128` is a power of two (the tree
reductions assume it), a multiple of the 32-lane warp, and comfortably covers the
small `L` (each thread strides). A second kernel does `Y = Z·Wo` with **one thread
per output element** (`grid = ceil(L·d / 256)`); a third computes each row norm
with **one thread per residue**.

**Memory hierarchy.**
- **Shared memory** holds the whole length-`L` logit/weight row (`s_logit`) plus a
  128-float reduction scratch (`s_red`). Dynamic shared size `= (L + 128)·4` bytes
  is set at launch. Keeping the row in shared memory means the softmax reductions
  never touch global memory.
- **Registers** hold each thread's `q`/`k` slices (`d_head ≤ 64`, capped).
- **Global memory** holds `X`, `Z`, `Y`, and the head-0 attention map.
- We recompute `Q/K/V` on the fly (via `proj_one`) instead of storing them — a
  compute-for-memory trade that keeps the teaching kernel small; a production
  kernel stages projected tiles in shared memory once.

**No CUDA library is used here** — the point is to *see* the GEMM + softmax. In
production these become **cuBLAS** GEMMs (`QKᵀ`, `AV`, `ZWo`) and a **cuDNN /
FlashAttention** fused-attention call. Writing those by hand at full performance
means tiling for shared memory and Tensor Cores, double-buffering, and the online
softmax trick — which is exactly what FlashAttention does (§7).

## 5. Numerical considerations

**Precision.** Tensors are FP32 (as in mixed-precision PLM inference, where the
*accumulators* are FP32). Every inner product — projections, scores, value blends,
norms — accumulates in **double** on both the CPU and the GPU, so the dominant
rounding is controlled and matched.

**Stability.** The softmax subtracts the row max before exponentiating
(`exp(s − m)`, argument ≤ 0), so it never overflows regardless of logit
magnitude. Both sides use the identical `softmax_inplace()` from
`attention_math.h`.

**Determinism & reductions.** The softmax **denominator** is summed in a different
*order* on the GPU (a parallel tree reduction over 128 partials) than on the CPU
(a left-to-right loop). Floating-point addition is not associative, so the two
sums differ in the last bits. Propagated through the pipeline (project → score →
softmax → blend → project) this yields a divergence of `~1e-8` in the output
embeddings — see the demo's `max_abs_err`. The result is nonetheless **fully
deterministic run-to-run** (the reduction order is fixed by the block size), so
the demo's stdout is byte-identical every time. The value blend and the row norms
are single-thread sequential sums, so those match the CPU even more tightly.

We deliberately do **not** use floating-point `atomicAdd` anywhere: it would make
the reduction order nondeterministic and the stdout irreproducible (PATTERNS.md
§3). All cross-thread accumulation is via ordered tree reductions in shared memory.

## 6. How we verify correctness

The CPU reference (`src/reference_cpu.cpp`, `attention_cpu`) recomputes the entire
block serially using the **same** `attention_math.h` primitives the kernel calls.
`main.cu` then checks three things:

1. `max_abs_err(out_cpu, out_gpu)` ≤ `1e-4` — the output embeddings agree.
2. `max_abs_err(attn_cpu, attn_gpu)` ≤ `1e-4` — the head-0 attention map agrees.
3. `top_attn_cpu == top_attn_gpu` — the discrete most-attended-residue readout is
   *identical*.

**Why `1e-4`?** This is the "long FP32 pipeline" tolerance from PATTERNS.md §4: a
multi-stage computation where the GPU's reduction order and fused multiply-adds
diverge from the host compiler's. The observed error is `~1.5e-8` — far tighter
than the bound — so the tolerance is honest headroom, not a fudge. We do not
pretend the two are bit-identical; we verify to a physically-negligible gap and
say so.

**Why this is convincing.** The CPU reference is an *independent* serial
implementation (different control flow, no shared-memory reductions, no threads).
When two independent implementations of the same math agree to 8 significant
figures across every residue *and* produce the identical discrete argmax readout,
a transcription or indexing bug in either would almost certainly have shown up.
Edge cases exercised: the ragged last block in the output-projection kernel (guard
`idx >= L·d`), threads with `j ≥ L` in the strided loops (they contribute `−FLT_MAX`
to the max and `0` to the sum, both harmless), and `d_head ≤ 64` (asserted on the
host before launch).

## 7. Where this sits in the real world

This is a **reduced-scope teaching version**. A production PLM differs in scale and
structure:

- **Trained weights.** ESM-2 ships hundreds of MB to tens of GB of *learned*
  parameters; ours are deterministic hashes with no biological signal. Swapping in
  real weights (and a real tokenizer with `<cls>/<eos>/<pad>/<mask>`) is the bridge
  to a working model — see fair-esm.
- **Positional encoding.** We use *none*, which is why identical residues get
  identical outputs in the demo. ESM-2 uses **rotary positional embeddings (RoPE)**
  that rotate Q/K by position, breaking that symmetry. (Exercise 1.)
- **Full Transformer block.** Real layers add residual connections, LayerNorm, and
  a feed-forward MLP, and stack ~33 of them. We isolate one attention sub-layer.
- **FlashAttention.** We materialize the full `L×L` attention row in shared memory
  — fine at `L = 24`, impossible at `L = 1024` × many heads. **FlashAttention**
  (Dao et al.) tiles the keys and maintains a *running* max and sum (the "online
  softmax"), never storing the full row, turning the memory cost from `O(L²)` to
  `O(L)` and fusing the three GEMMs. That is the single most important
  optimization separating this demo from a production kernel.
- **Tensor Cores & precision.** Production runs the GEMMs in FP16/BF16 on Tensor
  Cores with FP32 accumulation (`>3×` MFU on H100), via cuBLAS/cuDNN.
- **Parallelism & batching.** Inference over millions of proteins uses tensor +
  pipeline parallelism (Megatron-LM / DeepSpeed) and **dynamic batching by length
  bucket** so padding is minimized — the engineering challenge the catalog names.

---

## References

- **Vaswani et al., "Attention Is All You Need" (2017)** — scaled dot-product and
  multi-head attention; the math in §2.
- **Lin et al., "Evolutionary-scale prediction of atomic-level protein structure
  with a language model" (Science 2023)** — ESM-2 / ESMFold.
- **Dao et al., "FlashAttention" (2022)** — the tiled, memory-efficient attention
  kernel; the production version of our §4 (study §7 and Exercise 4).
- **fair-esm** (<https://github.com/facebookresearch/esm>) — reference ESM-2/ESMFold
  inference code and weights; how a real PLM is structured.
- **EvolutionaryScale ESM3** (<https://github.com/evolutionaryscale/esm>) — the
  latest multimodal protein model.
- **ColabFold** (<https://github.com/sokrypton/ColabFold>) — the MSA-based path
  ESMFold avoids; good contrast for *why* PLMs are faster on single sequences.
