# THEORY — 2.25 Coevolutionary Contact Prediction & MSA Transformer

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

A protein is a chain of amino acids that folds into a specific 3-D shape. The
shape is what does the work (catalysis, binding, signaling), so predicting it
from sequence is one of biology's central problems. A powerful clue hides in
**evolution**.

Consider two residues `i` and `j` that sit far apart along the chain but **touch
in the folded structure** (a "contact", say a salt bridge or a hydrophobic
packing pair). If a random mutation at position `i` would destabilize that
contact, the protein is less fit — unless a **compensating** mutation appears at
position `j` that restores the interaction. Over millions of years and millions
of descendant species, contacting positions therefore **mutate in a correlated
way**, while non-contacting positions vary independently.

We can read this correlation directly from a **Multiple Sequence Alignment
(MSA)**: take a protein, find thousands of its homologs (related sequences from
other organisms), and align them so that *equivalent* positions line up in
**columns**. Now each column is a position in the family, and each row is one
homolog. **Two columns that coevolve are statistically dependent** — knowing the
amino acid in column `i` tells you something about column `j`. Quantifying that
dependence for every column pair yields a **contact map**: a prediction of which
residues are close in 3-D. That map is exactly the constraint that lets methods
from EVcouplings (2011) to AlphaFold (2021) build accurate structures.

This project builds the **foundational, exact** version of that idea: pairwise
**Mutual Information** between MSA columns, cleaned with the **Average Product
Correction**. It is the conceptual seed of every coevolution method.

```
        MSA (N sequences x L columns)              Coevolution matrix (L x L)
        col: 1 2 3 ... i ......... j ... L
 seq1    M K T ... D ......... K ...        ====>      one MI value per pair (i,j)
 seq2    M R S ... E ......... R ...                   high MI  -> likely contact
 seq3    M K T ... K ......... D ...                   low  MI  -> likely no contact
  ...                ^             ^
                  column i      column j  <- these covary (D<->K, E<->R, K<->D):
                                             a coevolving (contacting) pair
```

## 2. The math

**Setup.** The MSA has `N` sequences and `L` columns. Each cell holds a symbol
from an alphabet of size `Q = 21` (the 20 amino acids `ACDEFGHIKLMNPQRSTVWY` plus
a gap `-`). For a column `i` let `f_i(a)` be the empirical frequency of symbol `a`:

```
f_i(a) = (1/N) * #{ sequences whose column i holds symbol a }       (a marginal)
```

For an ordered column pair `(i, j)` let `f_ij(a, b)` be the empirical **joint**
frequency:

```
f_ij(a,b) = (1/N) * #{ sequences with symbol a in col i AND symbol b in col j }
```

**Mutual Information** of the two columns (in *nats*, because we use the natural
log) is

```
              Q-1  Q-1                          f_ij(a,b)
  MI(i,j) =   sum  sum  f_ij(a,b) * ln( --------------------------- )
              a=0  b=0                      f_i(a) * f_j(b)
```

with the convention `0 · ln 0 = 0` (empty cells contribute nothing). Properties:

- `MI(i,j) ≥ 0`, with equality **iff** the two columns are statistically
  independent (`f_ij = f_i · f_j`) — i.e. no coevolution.
- `MI` is symmetric: `MI(i,j) = MI(j,i)`.
- It is the **Kullback–Leibler divergence** between the observed joint
  distribution and the product of marginals: "how far from independent are these
  two columns?"

**The bias problem and APC.** Raw MI has a systematic flaw: a column with high
**entropy** (lots of different amino acids) shares apparent information with
*every* other column, purely as a statistical/phylogenetic artifact — not because
of a real contact. The **Average Product Correction** (Dunn, Wuchty & Bonneau,
*Bioinformatics* 2008) estimates and removes this background:

```
  APC(i,j) = MI(i,j) - ( MI_col(i) * MI_col(j) ) / MI_mean

  MI_col(i) = mean over j != i of MI(i,j)        (column i's average coupling)
  MI_mean   = mean over all i != j of MI(i,j)    (the global average coupling)
```

The product term is large exactly when *both* columns are "bright on average", so
subtracting it suppresses the entropic background and leaves the **specific**
coevolution between `i` and `j`. APC turns mediocre raw MI into a respectable
contact predictor and is still applied on top of modern DCA scores.

**Output.** We rank all pairs `(i, j)` with `i < j` by `APC(i,j)`; the highest are
the predicted contacts.

## 3. The algorithm

```
1. Tokenize the MSA:  letters -> integers in [0, Q)              O(N*L)
2. Column marginals:  single[c][a] for every column c, symbol a  O(N*L)
3. For each column pair (i, j), i < j:                           O(L^2) pairs
     a. build joint counts pair[a][b] over the N sequences          O(N)
     b. MI(i,j) = cv_mi_from_counts(pair, single[i], single[j], N)   O(Q^2)
4. APC correction over the L x L MI matrix                        O(L^2)
5. Sort pairs by APC score, report the top-K                     O(L^2 log L)
```

**Complexity.** Step 3 dominates: `O(L^2 * (N + Q^2))`. The `N` term is the joint
counting (one pass over all sequences per pair), the `Q^2 = 441` term is the MI
reduction. For a real protein (`L ≈ 300`, `N ≈ 10^4`) that is ~`9·10^4` pairs ×
~`10^4` = ~`10^9` count-increments — heavy, but **embarrassingly parallel**: every
pair is independent.

**Data-access pattern.** Each pair reads two columns of the MSA (a strided gather:
column `i` is at offsets `i, i+L, i+2L, …`) and writes one scalar. Arithmetic
intensity is modest (counting + a log per non-empty joint cell), so on a CPU this
is memory- and branch-bound; on a GPU it becomes thousands of independent small
reductions.

## 4. The GPU mapping

**Pattern:** "score all pairs, each independent" (`docs/PATTERNS.md` §1;
exemplars `1.12` Tanimoto, `12.01` spectral search). We map **one column pair →
one thread**.

- **Thread-to-data map.** A 2-D grid of 2-D blocks. Thread
  `(i, j)` with `i = blockIdx.x·blockDim.x + threadIdx.x` and
  `j = blockIdx.y·blockDim.y + threadIdx.y` owns matrix cell `(i, j)`. Because MI
  is symmetric we do real work only for `i < j` and write **both** `mi[i,j]` and
  `mi[j,i]`; threads with `i ≥ j` (or out of range) return immediately.
- **Launch configuration.** Block = `16 × 16 = 256` threads (a multiple of the
  32-lane warp; 8 warps to hide latency). Grid = `ceil(L/16) × ceil(L/16)` so the
  whole `L × L` matrix is covered. A *square* block matches the square output.
- **Memory hierarchy.**
  - **Global memory:** the MSA `tokens` (`N·L` bytes, `uint8`) and the
    precomputed `single` marginals (`L·Q` `uint32`) live in global memory, read
    by the threads.
  - **Registers / local memory:** each thread keeps its own `Q·Q = 441`-entry
    `uint32` joint-count table (~1.7 KB) in per-thread local memory. This is the
    big simplification of the teaching version — it trades memory for clarity.
  - **No shared memory, no atomics:** threads write **disjoint** output cells, so
    there is nothing to synchronize and no contention. (A faster design would
    tile the MSA into shared memory and reuse it across a block's pairs — an
    exercise.)
- **Why precompute marginals on the host?** The `single` table is tiny and reused
  by all `O(L^2)` threads; recomputing it per thread would multiply the work by
  `L^2`. So we count marginals once on the CPU and upload them — the same "finish
  the cheap reduction on the host" idea flagship `11.09` uses for k-means
  centroids.
- **The APC step stays on the host.** It is a cheap `O(L^2)` reduction over the MI
  matrix; doing it on the host lets the **CPU and GPU share one `apc_correct()`**,
  guaranteeing identical corrected scores (no separate kernel to keep in sync).
- **No CUDA library here.** The catalog mentions cuBLAS (for the `L×L` coupling
  matrix products of mean-field DCA) and PyTorch axial attention (for the MSA
  Transformer). This teaching version needs neither: MI is a counting + reduction
  problem, not a dense linear-algebra one. §7 explains where those libraries enter
  in the production methods.

```
   Grid of 16x16 blocks tiles the L x L matrix of column pairs:

      j ->
    +----+----+----+ ...
  i | B00| B01| B02|         each cell = one thread = one MI(i,j)
  | +----+----+----+         threads with i >= j return (lower triangle wasted)
  v | B10| B11| B12|         every thread: read cols i,j -> count -> MI -> write
    +----+----+----+
    | ...                    block = (16,16) = 256 threads
```

## 5. Numerical considerations

- **Counting is exact.** All the heavy work is **integer** addition (joint counts,
  marginals). Integer adds **commute**, so the counts are bit-identical on CPU and
  GPU regardless of thread scheduling. There is no floating-point reduction whose
  order could vary.
- **Precision.** MI itself is accumulated in **`double`**. We deliberately iterate
  the `(a, b)` joint cells in the **same fixed order** (a outer, b inner) on host
  and device. Floating-point addition is not associative, but it *is*
  deterministic for a fixed order — so both sides produce the same bits up to the
  one place they can differ: `std::log`. The host C++ library's `log` and nvcc's
  device `log` may disagree by ~1 ulp.
- **No NaN/Inf traps.** Inside `cv_mi_from_counts`, a non-empty joint cell
  (`nab > 0`) implies both marginals are positive, so `ln(p_ab/(p_i·p_j))` always
  has a finite, positive argument. Empty cells are skipped (`0·ln 0 = 0`). APC
  guards `MI_mean == 0` (a fully-conserved MSA) to avoid a divide-by-zero.
- **Determinism (PATTERNS.md §3).** Because we use **no floating-point atomics**
  and a fixed summation order, stdout is byte-identical every run. Run-varying
  numbers (timings, the verify epsilon) go to **stderr**.

## 6. How we verify correctness

The CPU reference (`src/reference_cpu.cpp`) is written to be *obviously* correct:
a plain triple loop over column pairs, building joint counts and calling the
**same** `cv_mi_from_counts` the kernel calls. `main.cu` runs both paths on the
same MSA and compares the **raw MI matrices** element-by-element.

- **Tolerance:** `1e-9` nats (PATTERNS.md §4, "~machine precision"). Both sides
  compute identical integer counts and evaluate the same arithmetic in the same
  order; the only slack is the ~1-ulp `log` difference between the host and device
  math libraries. In practice the observed `max |MI_gpu − MI_cpu|` is **~4e-16**
  (true machine precision) — printed on the stderr `[verify]` line.
- **Why this is convincing.** The GPU kernel and the CPU reference are
  *independent* implementations (different loop structure, different memory model,
  different compiler for the device code). Agreement to machine precision across
  two independent codes is strong evidence the computation is right.
- **A stronger, scientific check.** Beyond CPU==GPU, the demo validates the
  *science*: the synthetic MSA has **four planted contacts** with known ground
  truth, and the method ranks all four at the top (APC ≈ 1.3–1.4) far above the
  best decoy (≈ 0.13). That confirms the pipeline recovers the signal it was
  designed to find, not just that two codes agree on a wrong answer.
- **Edge cases:** the loader rejects a non-rectangular (ragged) MSA; conserved
  columns (zero entropy) correctly yield ~0 MI; the APC denominator is guarded.

## 7. Where this sits in the real world

This project computes **pairwise** MI, which has a well-known limitation:
**indirect coupling**. If `i`–`j` and `j`–`k` are each in contact, then `i`–`k`
will *also* appear correlated even if they never touch — correlation is
transitive, contact is not. Production methods fix this by modeling all positions
**jointly**:

- **DCA / mean-field DCA.** Treat the MSA as samples from a Potts (21-state Ising)
  model and infer **direct** couplings `J_ij(a,b)` by (approximately) inverting
  the `LQ × LQ` covariance matrix. The inversion "explains away" indirect
  correlations. The `L×L` matrix products and the inverse are where **cuBLAS /
  cuSOLVER** earn their keep (the catalog's "cuBLAS for L×L coupling matrix
  products").
- **PLMC / CCMpred (pseudolikelihood DCA).** Instead of the expensive partition
  function, maximize the **pseudolikelihood** (each column predicted from all
  others) by gradient descent. CCMpred writes **custom CUDA kernels** for that
  per-position gradient — the same independent-pairs structure we use, but for a
  gradient instead of a count. EVcouplings (PLMC) is the reference implementation.
- **MSA Transformer (ESM-MSA-1b).** A deep network with **tied axial attention**:
  attention along rows (within a sequence) and along columns (across homologs),
  trained on millions of MSAs. Its column-attention maps *are* a learned
  coevolution signal, and reading contacts off them beats classical DCA,
  especially for shallow MSAs. This is the **PyTorch CUDA** route in the catalog.
- **Upstream MSA quality** (HHpred/HHblits, jackhmmer) and **sequence
  reweighting** matter as much as the estimator: a deeper, less redundant MSA
  improves *every* method.

Our MI+APC is the honest, exact, GPU-parallel *starting point* those methods build
on — and the cleanest place to learn the "score all column pairs" GPU pattern.

---

## References

- **Dunn, Wuchty, Bonneau (2008)**, *Bioinformatics* — introduces the **Average
  Product Correction**; the source of our `apc_correct`. Read for why raw MI needs
  background subtraction.
- **Morcos et al. (2011)**, *PNAS* — **Direct Coupling Analysis**; the foundational
  "direct vs. indirect" argument that motivates going beyond MI.
- **Ekeberg et al. (2013)**, *Phys. Rev. E* — **pseudolikelihood DCA** (plmDCA),
  the algorithm behind CCMpred/EVcouplings PLMC.
- **EVcouplings** — <https://github.com/debbiemarkslab/EVcouplings> — production
  DCA coevolution; study the PLMC pipeline and contact scoring.
- **CCMpred** — <https://github.com/soedinglab/CCMpred> — GPU DCA; study the custom
  CUDA gradient kernels (the per-pair parallelism, scaled up).
- **Rao et al. (2021)**, *ICML* / **ESM-MSA-1b** —
  <https://github.com/facebookresearch/esm> — the MSA Transformer; study tied
  axial attention as a learned generalization of column statistics.
- **HHpred / HHblits** — <https://toolkit.tuebingen.mpg.de/tools/hhpred> — building
  the deep, high-quality MSA every coevolution method depends on.
