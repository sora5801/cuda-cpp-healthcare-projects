# THEORY — 3.9 Phylogenetic Likelihood / Tree Inference

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

**Phylogenetics** reconstructs the evolutionary tree relating a set of biological
sequences (genes, genomes, proteins). Given an aligned set of DNA sequences — one
per *taxon* (species/sample), all the same length — we want the **tree** (branching
pattern + branch lengths) that best explains how those sequences diverged from a
common ancestor by mutation. Trees answer questions like "which species are most
closely related?", "when did two lineages split?", and "where did this virus
strain come from?" (phylodynamics of SARS-CoV-2, influenza, HIV all rest on exactly
this machinery).

The dominant rigorous criterion is **maximum likelihood (ML)**: treat mutation as a
random process running along the tree's branches, and score a candidate tree by the
**probability it assigns to the observed alignment**. The tree with the highest
likelihood is preferred. Computing that probability for one tree is what this
project does; searching over trees (RAxML, IQ-TREE) and integrating over them
(Bayesian MCMC, MrBayes) are layers built *on top* of this inner kernel.

The key modeling assumption that makes ML tractable: **sites (alignment columns)
evolve independently**. So the likelihood of the whole alignment factorises into a
product over sites — and that product is the source of the GPU parallelism.

## 2. The math

**Substitution model (continuous-time Markov chain).** DNA has 4 states
{A, C, G, T}. Mutation is modeled as a Markov process with a 4×4 instantaneous
**rate matrix** `Q`. Over a branch of length `t` (measured in *expected
substitutions per site*), the probability of ending in state `y` given start state
`x` is the `(x,y)` entry of the **transition matrix**

```
P(t) = exp(Q t)            (matrix exponential)
```

This project uses the **Kimura 2-parameter (K2P)** model, which splits substitutions
into **transitions** (purine↔purine A↔G, or pyrimidine↔pyrimidine C↔T) and
**transversions** (the other changes), with a rate ratio `kappa` (transitions are
`kappa`× as likely; `kappa = 1` gives Jukes–Cantor). K2P's `P(t)` has a **closed
form** (no numerical exponential needed):

```
p_same(t) = 1/4 + 1/4 e^{-4 b t} + 1/2 e^{-2(a+b) t}     (x -> x)
p_ts(t)   = 1/4 + 1/4 e^{-4 b t} - 1/2 e^{-2(a+b) t}     (a transition)
p_tv(t)   = 1/4 - 1/4 e^{-4 b t}                          (each transversion)
```

with rates `b = 1/(kappa+2)` (transversion) and `a = kappa·b` (transition),
normalised so `a + 2b = 1` (one transition partner + two transversion partners per
base) — hence `t` is in substitutions/site. This is `k2p_prob()` in
[`src/felsenstein.h`](src/felsenstein.h).

**Site likelihood (the objective).** For one site, let `L_k[s]` be the *conditional
likelihood* = P(all data in the subtree below node `k` | node `k` is in state `s`).
**Felsenstein's pruning recursion** computes it bottom-up:

- **Leaf** `l` with observed base `b`: `L_l[s] = 1` if `s = b`, else `0` (a gap/`N`
  gives `L_l[s] = 1` for all `s` — uninformative).
- **Internal node** `k` with children `u` (branch `t_u`) and `v` (branch `t_v`):

```
L_k[s] = ( Σ_x P(s->x, t_u) · L_u[x] ) · ( Σ_y P(s->y, t_v) · L_v[y] )
```

The site likelihood combines the root CLV with the equilibrium frequencies
`π_s = 1/4` (K2P):  `L(site) = Σ_s π_s · L_root[s]`. The **tree log-likelihood** is

```
lnL(tree) = Σ_sites ln L(site)
```

Inputs: the alignment (states per taxon per site), the tree (topology + branch
lengths), `kappa`. Output: one real number `lnL` per tree (always ≤ 0). Higher is
better. The argmax over candidate trees is the ML tree.

## 3. The algorithm

For each candidate tree, for each site, run the pruning recursion:

```
score_tree(tree, alignment):
    total = 0
    for each site j:                         # INDEPENDENT  -> GPU parallel axis
        for each leaf l:  L_l = onehot(obs[j][l])          # base case
        for each internal node k in POST-ORDER:            # children before parent
            for s in {A,C,G,T}:
                L_k[s] = (Σ_x P(s,x,t_left)·L_left[x])
                       · (Σ_y P(s,y,t_right)·L_right[y])
        total += ln( Σ_s (1/4)·L_root[s] )
    return total
```

**Post-order** (children stored before parents) lets a single forward sweep over the
node array visit every node only after its children are done — no recursion, no
stack. The loader validates this invariant (a child index must be a leaf or an
*earlier* internal node).

**Complexity.** A rooted binary tree on `n` taxa has `n−1` internal nodes. Each node
costs `NSTATES² = 16` multiply-adds. So **per site**: `O(n · 16) = O(n)` work.
**Per tree**: `O(n · n_sites)`. The serial cost is dominated by the `n_sites`
factor; the depth along the critical path is the tree height `O(log n)` for a
balanced tree, but because **sites are independent**, the *parallel* work has depth
`O(log n)` and width `n_sites` — exactly the shape a GPU wants.

**Arithmetic intensity.** Each site reads `n` bytes (its column) + the shared tree,
and does `O(n·16)` flops — compute-bound once `n` is modest, which is why the
per-site kernel scales well with alignment length.

## 4. The GPU mapping

**Thread-to-data map.** One **thread per alignment site**: site
`j = blockIdx.x · blockDim.x + threadIdx.x`. Thread `j` runs the entire pruning
recursion for column `j` (calling the *same* `site_log_likelihood()` the CPU uses)
and contributes one term to the tree's total.

**Launch configuration.** `block = 256` threads (8 warps — enough to hide global-
memory latency, a multiple of 32, good occupancy on sm_75…sm_89);
`grid = ceil(n_sites / 256)` blocks; the ragged last block is guarded by
`if (j >= n_sites) return;`.

**Memory hierarchy and why:**

- **Constant memory** holds the tree (`c_nodes[]`). Every thread reads the same
  nodes and never writes them → the constant cache **broadcasts** one node to a
  whole warp in a single transaction. A tree on ≤ 64 taxa is ~2 KB, far inside the
  64 KB constant bank.
- **Global memory, column-major** holds the alignment. Storing site `j`'s states
  contiguously means thread `j` reads a contiguous run → **coalesced** loads across
  the warp's `n_taxa` bytes.
- **Global scratch** holds each thread's conditional-likelihood vectors (CLVs):
  thread `j` owns the slice `clv[j·stride … ]` with `stride = (n_taxa+n_internal)·4`
  doubles, so threads never collide. We keep CLVs in global (not registers/shared)
  because the buffer size depends on `n_taxa` at runtime — a clean, always-correct
  teaching choice. For a *fixed small* `n_taxa` you would instead keep the CLVs in
  registers or per-block **shared memory** (an exercise) for a large speedup.

**No CUDA library is needed here.** K2P's transition matrix is closed-form, so we
never call cuBLAS/cuSOLVER for a matrix exponential. Production libraries (BeagleLib)
*do* use cuBLAS-style 4×4/20×20 matrix–vector products per site per node for general
GTR / amino-acid models; §7 explains what hand-rolling that entails.

```
                 grid of blocks (cover all sites)
   block 0                block 1                 ...
 ┌───────────────┐     ┌───────────────┐
 │ t0 t1 ... t255│     │t256 ...   t511│   each thread tj:
 └───────────────┘     └───────────────┘     reads column j  (global, coalesced)
        │                     │               reads tree      (constant, broadcast)
        ▼                     ▼               runs pruning -> site lnL
   site_log_likelihood per thread             atomicAdd(to_fixed(lnL)) -> int64 total
        └─────────────┬───────┘
                      ▼
        one fixed-point accumulator  (deterministic integer sum)
```

## 5. Numerical considerations

- **Precision: FP64.** Likelihoods are products of many small probabilities; we
  carry CLVs and the per-site `ln L` in `double`. For very deep trees, CLVs can
  underflow — production tools rescale CLVs periodically and track the log of the
  scaling factor. Our teaching trees are shallow, so plain FP64 suffices; rescaling
  is noted as the production fix.
- **The matrix `P(t)` is a stochastic matrix:** each row sums to 1 and entries are
  in `(0,1]`, so the recursion never produces negatives. `kappa ≥ 0`, `t ≥ 0`.
- **Determinism — the crux.** The GPU sums `n_sites` per-site log-likelihoods. A
  **floating-point `atomicAdd`** would add them in nondeterministic thread order;
  because FP addition is *not* associative, the total would wobble in the last bits
  run-to-run and would **not** equal the CPU's sum. Fix (PATTERNS.md §3): convert
  each per-site `lnL` to a **fixed-point integer** (`llround(lnL · 1e6)`) and
  `atomicAdd` *that* into a 64-bit accumulator. **Integer addition commutes**, so
  the total is independent of order — deterministic *and* bit-identical to the CPU,
  which sums the same integers. The scale `1e6` keeps 6 fractional digits;
  `|lnL_site| < ~50` over ≤ 10⁶ sites stays far inside int64's ±9.2×10¹⁸ range.
- **`llround` host/device parity.** `llround` rounds half-away-from-zero identically
  on the host and the device, so the per-site integer is the same on both sides.

## 6. How we verify correctness

Two independent checks (PATTERNS.md §4 + §6):

1. **GPU == CPU, exactly.** [`src/reference_cpu.cpp`](src/reference_cpu.cpp) runs the
   same `site_log_likelihood()` serially and reduces in the **same fixed-point
   integers**. Because both sides sum identical integers, the per-tree totals are
   **bit-identical** — `main.cu` verifies `max |lnL_cpu − lnL_gpu| ≤ 0.5/scale`
   (i.e. less than half a fixed-point ULP; on the sample it is exactly `0`). An
   independent serial reimplementation agreeing to the last bit is strong evidence
   the GPU kernel is correct.
2. **The science check.** The committed sample is DNA **simulated down a known
   tree** (`scripts/make_synthetic.py`). The program recovers that true tree as the
   maximum-likelihood winner (its `lnL` is highest), while the wrong NNI
   rearrangements score ~900–1100 log-units worse. So we validate not just
   "CPU==GPU" but "the method finds the right answer".

**Edge cases handled:** gaps/`N` (CLV all-ones), the degenerate all-zero site
likelihood (guarded against `log(0)`), and the post-order/child-index invariant
(rejected at load time with a clear message).

## 7. Where this sits in the real world

This is a **reduced-scope teaching version**. Production ML phylogenetics adds:

- **General substitution models.** GTR (DNA, 6 rates + base frequencies) and
  amino-acid models (WAG, LG; 20 states, 20×20 matrices) have **no closed-form**
  `exp(Qt)`. Tools eigendecompose `Q = U Λ U⁻¹` once, then
  `P(t) = U diag(e^{λ_i t}) U⁻¹` per branch — this is where cuBLAS/cuSOLVER (or
  BeagleLib's custom 4×4/20×20 kernels) earn their keep. Hand-rolling it means an
  eigensolver plus a batched GEMM per branch length.
- **Rate heterogeneity.** A Γ distribution of site rates (usually 4 discrete
  categories) multiplies the per-site work by the category count.
- **Branch-length & parameter optimisation.** Newton/BFGS re-optimises every branch
  length (and `kappa`, base frequencies) *per topology* — the likelihood here is
  the inner function those optimisers call thousands of times.
- **Tree search.** NNI and SPR moves propose new topologies; RAxML-NG / IQ-TREE
  hill-climb over millions of them. Each move re-scores the tree — exactly this
  kernel, which is why GPU acceleration of the site likelihood (BeagleLib, reported
  ~63×) matters so much.
- **Bayesian inference.** MrBayes / BEAST run Metropolis–Hastings MCMC, each step a
  full-tree likelihood; BeagleLib provides the CUDA-accelerated site likelihood
  under both.
- **CLV rescaling, multi-GPU partitioning, SIMD/AVX CPU paths** — engineering for
  numerical robustness and scale that we omit for clarity.

---

## References

- **Felsenstein, J. (1981).** "Evolutionary trees from DNA sequences: a maximum
  likelihood approach." *J. Mol. Evol.* — the pruning recursion implemented here.
- **Kimura, M. (1980).** The 2-parameter model whose closed-form `P(t)` we use.
- **BeagleLib** (https://github.com/beagle-dev/beagle-lib) — the GPU phylogenetic
  likelihood library behind MrBayes/BEAST; study its per-site/per-node kernels and
  CLV rescaling. *We reimplement the idea didactically; we do not copy its code.*
- **RAxML-NG** (https://github.com/amkozlov/raxml-ng) — fast ML inference; see how
  the likelihood inner loop drives NNI/SPR search.
- **IQ-TREE 2** (https://iqtree.github.io/) — model selection + ML search; a good
  reference for GTR/Γ and the parameter-optimisation layer we omit.
- **MrBayes** (https://github.com/NBISweden/MrBayes) — Bayesian MCMC phylogenetics,
  the canonical consumer of CUDA site-likelihood acceleration.
