# THEORY — 7.3 Clinical NLP over Notes & Records

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

This project ships a **reduced-scope teaching version** (CLAUDE.md §13) of clinical
NLP: it implements **one transformer self-attention encoder block** — the
computational heart of every BERT-style clinical model — and nothing else (no
training, no tokenizer, no task head). Everything below explains that block, then
§7 describes the full production system it lives inside.

## 1. The science

Hospitals generate enormous volumes of **free-text** clinical documentation:
discharge summaries, radiology and pathology reports, nursing and progress notes.
Buried in that prose are structured facts a computer could use — diagnoses, drugs,
lab findings, procedures, temporal relations, and the coreference chains that tie
"the patient", "he", and "Mr. Jones" to one entity. **Clinical NLP** extracts those
facts: named-entity recognition (NER), relation extraction (RE), ICD billing-code
assignment, phenotyping, and clinical-event prediction.

Since ~2019 the dominant approach is the **transformer**: a neural network that
represents each token (word-piece) as a vector and repeatedly lets tokens exchange
information via **self-attention**. Models like BioClinicalBERT and GatorTron are
*pretrained* on billions of clinical tokens, then *fine-tuned* on a specific task.
The single operation that makes a transformer a transformer — and that dominates
its cost — is self-attention. Understanding it is the point of this project.

**The concrete question this demo answers:** given a note, *which other tokens does
each token pay attention to?* We plant a coreference-like signal (the pronoun `he`
is built to resemble `patient`) and check that a correct attention block makes `he`
attend to `patient`. That is a toy of the coreference task the i2b2/n2c2 challenges
score.

## 2. The math

A note is a sequence of `S` tokens. Embed each token as a row of `X ∈ ℝ^{S×D}`
(`D` = model dimension). Self-attention transforms `X` into a contextualized
`O ∈ ℝ^{S×D}` as follows.

**Projections.** Three learned matrices `Wq, Wk, Wv ∈ ℝ^{D×D}` produce queries,
keys, and values:

```
Q = X·Wq      K = X·Wk      V = X·Wv          (each S×D)
```

**Multi-head split.** Partition the `D` columns into `H` heads of width
`dh = D/H`. Head `h` uses columns `[h·dh, (h+1)·dh)` of `Q`, `K`, `V`, denoted
`Q_h, K_h, V_h ∈ ℝ^{S×dh}`.

**Scaled dot-product attention.** For each head, the affinity of query token `i`
for key token `j` is their dot product, scaled:

```
scores_h = (Q_h · K_hᵀ) / √dh                 (S×S)
```

The `1/√dh` scaling keeps the dot products from growing with `dh` (which would
saturate the softmax). **Masking:** if key `j` is a `[PAD]` position, set
`scores_h[i,j] = −∞` so it gets zero probability.

**Softmax.** Each row becomes a probability distribution over keys:

```
A_h[i,j] = exp(scores_h[i,j]) / Σ_k exp(scores_h[i,k])          (rows sum to 1)
```

**Context.** Mix the value vectors by those weights, then concatenate heads:

```
O_h = A_h · V_h            (S×dh)
O   = concat(O_1, …, O_H)  (S×D)
```

Symbols: `S` sequence length, `D` model dim, `H` heads, `dh = D/H` head width,
`B` batch size (notes). All arithmetic here is `double` (FP64) so CPU and GPU can
agree tightly (§5).

## 3. The algorithm

Serial pseudocode for the whole batch:

```
for each note b in 0..B:
    X = gather embeddings of note b            # S×D
    Q,K,V = X·Wq, X·Wk, X·Wv                    # 3 × (S×D×D) matmuls
    for each head h:
        for each query i:
            for each key j: scores[j] = (Q_h[i]·K_h[j])/√dh, or −∞ if j is PAD
            A[i,:] = softmax(scores)            # stable, max-subtracted
            O_h[i] = Σ_j A[i,j]·V_h[j]           # S×dh context
```

**Complexity.** Per note: projections cost `O(S·D²)`; attention costs
`O(H·S²·dh) = O(S²·D)`. The `S²` term is the famous **quadratic-in-sequence-length**
cost: doubling the note length quadruples the attention work and memory. For a
long discharge summary (thousands of tokens) this dominates — the exact pain point
the catalog deep-dive and Flash Attention address (§7).

**Arithmetic intensity.** The two attention matmuls and the three projections are
dense GEMMs — high arithmetic intensity, compute-bound at scale, and a perfect fit
for GPU tensor cores. The softmax is memory-bound (one pass of reads/writes per
row). This split — GEMM, then a light reduction, then GEMM — is why we map the
GEMMs to cuBLAS and hand-write only the softmax.

## 4. The GPU mapping

We split the block into five GPU stages (see `src/kernels.cu`):

1. **`build_projections_kernel`** — one thread per entry of the packed
   `[3·D×D]` `Wq|Wk|Wv` buffer, each evaluating the shared `attn::proj_entry`
   recipe (identical to the CPU, so weights match bit-for-bit).
2. **`gather_kernel`** — one thread per `(token, dim)` of `X [(B·S)×D]`; an
   indirect load `X[row,d] = embed[ids[row]·D + d]` (the **gather** pattern).
3. **Projections `Q=X·Wq, K=X·Wk, V=X·Wv`** — three cuBLAS `Dgemm` calls over the
   whole batch (`(B·S)×D · D×D`).
4. **Scores `Q_h·K_hᵀ` and context `A_h·V_h`** — two `cublasDgemmStridedBatched`
   calls, one launch per note over its `H` heads (constant column stride `dh`).
5. **`scale_and_mask_kernel` + `softmax_kernel`** — element-wise scale/mask, then
   **one block per attention row** doing cooperative max and sum reductions in
   shared memory (the stable softmax of `attn_core.h`).

**Thread-to-data maps.** Stages 1/2/scale: linear index → decomposed coordinates
(guarded ragged last block). Softmax: `blockIdx.x` = the `(note,head,query)` row;
threads stride over the `S` keys and reduce through a shared-memory tree.

**Launch config.** The helper kernels use 256 threads/block (a warp multiple, good
occupancy on sm_75–sm_89). The softmax uses one block of 256 threads per row with
`256·sizeof(double)` bytes of shared memory for the two reductions.

**Memory hierarchy.** Global memory holds `X, Q, K, V`, the scores, and `O`.
Shared memory holds the softmax reductions. The projection weights live in global
memory (tiny). At this teaching scale everything is launch/copy-bound; the point is
the *mapping*, not the speed (§ honest-timing, PATTERNS.md §7).

**Row-major ↔ column-major (the load-bearing detail).** cuBLAS is column-major;
our arrays are row-major. A row-major `[m×n]` buffer read as column-major *is* its
`[n×m]` transpose, so a row-major `C = A·B` is obtained by asking cuBLAS for
`Cᵀ = Bᵀ·Aᵀ` — swap the operands and the `m/n` extents. Each call site in
`kernels.cu` states the exact `op`, leading-dimension, and batch-stride choice; the
`dgemm_rowmajor` helper wraps the projection case. **cuBLAS is not a black box
here** (CLAUDE.md §6.1.6): the comments spell out what `Dgemm` computes
(`C = αAB + βC`), the layout it expects, and that hand-rolling a competitive GEMM
means shared-memory tiling + register blocking + bank-conflict-free loads — which
is exactly why we delegate it.

```
per note b, per head h:                     softmax_kernel: 1 block per row
   Q_h [S×dh] ─┐                              row = (b,h,qi)
               ├─ cublasDgemmStridedBatched   ┌───────────── shared[] ──────────┐
   K_h [S×dh] ─┘        → scores [S×S]         │ reduce max → reduce Σexp → norm │
   scores [S×S] ─┐                             └─────────────────────────────────┘
                 ├─ cublasDgemmStridedBatched
   V_h [S×dh] ───┘        → O_h [S×dh]
```

## 5. Numerical considerations

- **Precision: FP64.** We use `double` throughout so the CPU reference and the GPU
  agree to ~machine precision. A production model uses FP16/BF16 with FP32
  accumulation on tensor cores; that is faster but noisier, and it would force a
  much looser verification tolerance.
- **Softmax stability.** `exp(x)` overflows for large `x`, so we subtract the row
  maximum first (`attn::softmax_exp`). This is algebraically identical but never
  exponentiates a positive argument. The `[PAD]` mask uses a large finite sentinel
  (`−1e30`) rather than a true `−∞` so the max-subtract arithmetic stays defined.
- **Determinism.** stdout is byte-identical every run: all reported numbers are
  FP64 results of fixed-order computations. The softmax reduction uses a
  fixed-shape shared-memory tree (deterministic within a run). We do **not** use
  `atomicAdd` on floats anywhere, so there is no float-reordering nondeterminism
  (PATTERNS.md §3). Timings go to stderr (not diffed).
- **FMA / summation order.** cuBLAS sums each dot product in a different order than
  the CPU's serial loop and uses fused multiply-add, so the two differ by ~`1e-16`
  here — a real, teachable non-associativity effect. It stays tiny because `dh` and
  `S` are small; it would grow with the head dimension.

## 6. How we verify correctness

`src/reference_cpu.cpp` implements the identical block in plain serial loops, using
the **same** `attn_core.h` per-element math the kernels use. `main.cu` runs both and
compares **two** arrays entrywise:

- the attention probabilities `A` `[B·H·S·S]` (tolerance `1e-11`), and
- the output embeddings `O` `[B·S·D]` (tolerance `1e-11`).

Observed worst differences are ~`1e-16`/`1e-16`, far under tolerance. The
tolerances are set to "double-precision, short computation" (PATTERNS.md §4): both
sides run the same formulas, so the only gap is FMA + summation order.

Why this is convincing: the CPU version is written to be *obviously* correct
(textbook triple loops, no cleverness), and it was authored independently of the
cuBLAS argument juggling. Agreement between an independent serial implementation and
the GPU pipeline is strong evidence the tricky column-major GEMM arguments are
right. A **second, science-level** check backs the numeric one: the demo confirms
the planted `he→patient` coreference link is recovered in all notes — validating
that the block does the *right computation*, not just that CPU==GPU.

Edge cases exercised by the sample: a note shorter than `S` (padding + masking), the
`[CLS]` summary token at position 0, and multiple heads (`H=2`).

## 7. Where this sits in the real world

A production clinical transformer wraps this block in much more:

- **Full encoder layer.** Each layer adds a residual connection + LayerNorm around
  attention, then a position-wise feed-forward network (two big GEMMs) with its own
  residual+norm. A model stacks 12–24 such layers.
- **Positional information.** Real models add positional encodings; modern ones
  (Clinical ModernBERT) use **Rotary Positional Embeddings (RoPE)**, rotating `Q`/`K`
  by position-dependent angles — omitted here (an exercise).
- **Flash Attention.** Materializing the `S×S` scores is the memory bottleneck at
  long context. **Flash Attention** fuses the score-GEMM, softmax, and context-GEMM
  into a single kernel that streams over key blocks keeping a running max and sum
  (online softmax), so it never stores the `S×S` matrix — dropping memory from
  `O(S²)` to `O(S)` and enabling 8192-token clinical notes. Our five-stage,
  scores-in-global design is the pedagogical opposite; exercise 4 sketches the fuse.
- **Training.** BERT is pretrained with **masked language modelling** (predict
  held-out word-pieces) on billions of clinical tokens, then fine-tuned per task
  (BIO-tagging + a CRF for NER, span-pair classifiers for RE, multi-label heads for
  ICD). We do none of this — our weights are fabricated, not learned.
- **Scale.** GatorTron-scale pretraining is **data-parallel across many A100/H100
  GPUs** with **NCCL** all-reduce and gradient checkpointing to fit long contexts in
  VRAM — the multi-GPU pattern the catalog names, well beyond a single-GPU teaching
  demo.
- **De-identification & governance.** Real notes are PHI; use is gated by
  credentialed access and data-use agreements (see `data/README.md`). Nothing here
  touches real patient text.

---

## References

- Vaswani et al., *Attention Is All You Need* (2017) — the transformer and scaled
  dot-product attention derived in §2.
- Devlin et al., *BERT* (2019) — masked-LM pretraining and the `[CLS]` pooling
  convention we mimic.
- Alsentzer et al., *Publicly Available Clinical BERT Embeddings* (BioClinicalBERT,
  2019) — clinical pretraining on MIMIC; the canonical starting model.
- Dao et al., *FlashAttention* (2022) — the fused, memory-linear attention kernel
  described in §7; the "why" behind the O(n²)→O(n)-memory claim.
- Yang et al., *GatorTron* (2022) — large-scale multi-GPU clinical LM pretraining.
- Clinical ModernBERT (<https://github.com/Simonlee711/Clinical_ModernBERT>) — RoPE +
  long-context clinical encoding.
- NVIDIA cuBLAS documentation — `cublasDgemm` / `cublasDgemmStridedBatched`
  semantics and the column-major convention used in §4.
