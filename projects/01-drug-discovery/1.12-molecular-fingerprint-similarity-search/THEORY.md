# THEORY — 1.12 Molecular Fingerprint Similarity Search

> Written for a reader who knows C++ but is new to CUDA and to cheminformatics.
> See [README.md](README.md) for the quick tour and build steps. _Educational only._

## 1. The science

Chemists routinely ask: *"which molecules in my library are most similar to this
one?"* — to find alternative drug candidates, to rationalize bioactivity ("similar
molecules tend to have similar properties"), or to deduplicate enormous virtual
libraries. To answer it at scale we need (a) a way to turn a molecule into a
comparable object, and (b) a similarity measure.

A **molecular fingerprint** is that object: a fixed-length bit string where each
bit marks the presence of a particular substructure. **Morgan / ECFP** (Extended
Connectivity FingerPrints) enumerate the circular atom environments up to a given
radius and hash each into a bit of a 2048-bit vector. Two molecules sharing many
substructures share many set bits.

## 2. The math

Let `A, B ∈ {0,1}^m` be two fingerprints (here `m = 2048`). The **Tanimoto
coefficient** (identical to the **Jaccard index** for sets) is

```
            |A ∧ B|        popcount(A AND B)
T(A,B) = ------------- = -----------------------
            |A ∨ B|        popcount(A OR  B)
```

where `popcount(x)` is the number of 1-bits. `T ∈ [0,1]`: `T=1` iff the
fingerprints are identical, `T=0` iff they share no bits. Note the identity
`|A ∨ B| = |A| + |B| − |A ∧ B|`, so a single pass computing the AND-popcount and
OR-popcount per word suffices.

**Problem statement.** Given a query `q` and library `L = {b_0, …, b_{N-1}}`,
compute `s_i = T(q, b_i)` for all `i`, and return the indices of the `K` largest.

## 3. The algorithm

```
for i in 0..N-1:                      # over library molecules  (PARALLEL)
    inter = 0; uni = 0
    for w in 0..FP_WORDS-1:           # over 64-bit words       (unrolled)
        inter += popcount(q[w] & b_i[w])
        uni   += popcount(q[w] | b_i[w])
    s_i = inter / uni
topK = indices of the K largest s_i
```

**Complexity.** Scoring is `Θ(N · FP_WORDS)` integer ops — `Θ(N)` for fixed width.
Serial on a CPU this is `N · 32` word-pairs, each ~2 popcounts. The top-K step is
`Θ(N log K)` with a partial sort. The scoring dominates for large `N`, and its
`N` independent iterations are exactly what we hand to the GPU.

## 4. The GPU mapping

**Decomposition.** One thread owns one library molecule. With block size
`B = 256` and `N` molecules we launch `ceil(N/B)` blocks (capped, see below).
Thread `(blockIdx.x, threadIdx.x)` starts at `i = blockIdx.x·B + threadIdx.x`.

```
  library (row-major, N x 32 words)         threads
  ┌───────────────────────────────┐         t0 -> row 0
  │ b0:  w0 w1 ... w31             │         t1 -> row 1
  │ b1:  w0 w1 ... w31             │         t2 -> row 2
  │ ...                            │          .
  │ bN-1:w0 w1 ... w31             │         t(N-1) -> row N-1
  └───────────────────────────────┘
        query q (32 words) lives in __constant__ memory, read by ALL threads
```

**Memory hierarchy — the two teaching points:**

- **Constant memory for the query.** Every thread reads all 32 query words and
  none writes them, and they are identical for the whole launch. `__constant__`
  memory has a dedicated cache that **broadcasts** one address to an entire warp
  in a single transaction. Putting `q` there (via `cudaMemcpyToSymbol`) avoids 32
  redundant global loads per thread. (256 bytes — trivially within the 64 KB
  constant bank.)
- **`__popcll` intrinsic.** 64-bit population count compiles to the hardware
  `POPC` instruction — one instruction per word, versus a multi-step bit-twiddle.

**Grid-stride loop.** Rather than require one thread per molecule, the kernel
loops `for (i = tid; i < n; i += gridDim.x*blockDim.x)`. This lets a **fixed,
modest grid** (we cap at 1024 blocks) cover a library of *any* size, keeps all SMs
busy, and is the idiomatic CUDA pattern for "map over a big array."

**Occupancy & bandwidth.** Each thread reads 32 × 8 = 256 bytes of library data
and does ~64 popcounts + 64 boolean ops — this is **memory-bound** at scale, so
performance tracks global-memory bandwidth. Block size 256 gives 8 warps/block,
enough to hide memory latency; registers and shared memory are not limiting (no
shared memory is used). Coalescing: consecutive threads read consecutive *rows*
(stride 32 words), so a warp's word-`w` accesses are 32×8 = 256 bytes apart —
**Exercise 3** (transposing the library to column-major) explores improving this.

**Which library does what.** This teaching version hand-rolls the kernel. In
production: **Thrust** `device_vector` holds the library; **cub::DeviceRadixSort**
does the on-device top-K; **Faiss** provides approximate (LSH/IVF) search when
exact brute force is too slow. We keep top-K on the host because, at the sizes a
learner runs, the scoring kernel is the whole story.

## 5. Numerical considerations

- **Precision.** `inter` and `uni` are integers in `[0, 2048]`, exactly
  representable as `float`. The only floating-point op is the final division.
- **Determinism.** There is **no reduction across threads** — each thread writes
  its own `out[i]` independently — so there is no atomic reordering and no
  floating-point non-determinism. CPU and GPU perform the *same* integer
  popcounts and the *same* IEEE-754 single-precision division (fast-math is OFF in
  the `.vcxproj`), so the results are **bit-identical** (`max_abs_err = 0`).
- **Edge case.** Two all-zero fingerprints give `0/0`; we define that as `0` to
  avoid `NaN` (both implementations guard it identically).

## 6. How we verify correctness

`main.cu` runs `tanimoto_cpu` (an obviously-correct serial loop using Kernighan's
popcount) and `tanimoto_gpu`, then reports `max_abs_err = max_i |s_i^{cpu} −
s_i^{gpu}|`. The demo asserts it is within `1e-6`; in practice it is exactly `0`.
Agreement between two independent implementations — one trivial, one parallel — is
strong evidence the kernel is correct. The committed sample is engineered so
similarities span the whole `[0,1]` range, exercising the division and the top-K
ordering (including ties, which break by lower index).

## 7. Where this sits in the real world

`gpusimilarity` and FPSim2 do the same core computation but add: streaming
libraries larger than GPU memory, multi-GPU sharding, on-device top-K, and
folding/unfolding of sparse fingerprints. Approximate methods (LSH, Faiss IVF)
trade exactness for sub-linear query time on billion-scale sets. The chemistry
upstream — generating good fingerprints (ECFP radius, bit count, counts vs.
binary) — matters as much as the search itself, and is handled by RDKit.

## References

- Rogers & Hahn, *Extended-Connectivity Fingerprints*, J. Chem. Inf. Model. 2010 — the ECFP definition.
- Schrödinger **gpusimilarity** — production CUDA brute-force Tanimoto search (the direct analogue).
- **RDKit** docs, *Morgan Fingerprints* — how the bit-vectors we consume are built.
- NVIDIA CUDA C++ Programming Guide — constant memory, intrinsics (`__popcll`), grid-stride loops.
