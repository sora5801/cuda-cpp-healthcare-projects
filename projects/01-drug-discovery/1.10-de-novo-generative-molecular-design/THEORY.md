# THEORY — 1.10 De Novo Generative Molecular Design

> **Reduced-scope teaching version.** Production de-novo design uses deep
> generative networks (RNN/transformer language models over SMILES, or diffusion
> models over 3-D molecular graphs) trained for days on multi-GPU nodes. Those
> cannot be a single deterministic CUDA demo. This project teaches the *concept
> and the GPU pattern* with the simplest model that still exhibits the whole
> pipeline — a **first-order Markov language model over SMILES characters** — and
> is honest about the gap (see §"Where this sits in the real world").

---

## 1. The science

**De novo molecular design** means inventing brand-new chemical structures that
optimise a goal (binding potency, selectivity, ADMET, synthesizability) instead
of searching an existing library. The modern recipe is *generative*:

1. **Represent molecules as strings or graphs.** The dominant string format is
   **SMILES** (Simplified Molecular-Input Line-Entry System): a molecule is
   written as a short sequence of characters — `CCO` is ethanol, `c1ccccc1` is
   benzene, `CC(=O)O` is acetic acid. Atoms are letters (`C`, `O`, `N`; lowercase
   for aromatic), bonds are `= #`, branches are parenthesised, and digits mark
   ring closures.
2. **Learn the distribution** of "drug-like" molecules from a corpus (ChEMBL,
   ZINC, MOSES). A generative model that has learned this distribution can
   *sample* novel strings that look like plausible molecules — this is
   **distribution learning**.
3. **Optimise toward a goal.** Fine-tune the sampler (e.g. with reinforcement
   learning) so the molecules it emits score highly on an objective — this is
   **goal-directed generation**. Each candidate is passed through a **scoring
   function** (docking score, QED drug-likeness, SA synthetic-accessibility), and
   the reward steers the next round of sampling.

A SMILES generative model is fundamentally an **autoregressive language model**:
it defines `P(next character | characters so far)` and builds a molecule one
token at a time, exactly like a text language model builds a sentence. That is
the single idea this project makes concrete.

---

## 2. The math

### 2.1 The generative model as a conditional distribution

An autoregressive model factorises the probability of a string
`s = (s₁, s₂, …, s_L)` as a product of next-token conditionals:

```
P(s) = ∏_{t=1..L}  P(s_t | s_1, …, s_{t-1})
```

A transformer/RNN models the full history `s_1…s_{t-1}`. Our **first-order
Markov** simplification truncates the history to just the previous token:

```
P(s_t | s_1, …, s_{t-1}) ≈ P(s_t | s_{t-1})          (Markov assumption)
```

So the entire model is a single **transition matrix** `T` of size `K×K` where
`K = NSYM` is the alphabet size (here 14, including a START/END sentinel):

```
T[a, b] = P(next symbol = b | current symbol = a)
```

### 2.2 Training = counting (maximum likelihood with smoothing)

The maximum-likelihood estimate of `T[a,b]` is just the normalised count of how
often `b` followed `a` in the training corpus. We frame every training string
with the sentinel `^` on both ends, `^ s₁ s₂ … s_k ^`, so the model also learns
which symbols **start** a molecule (transitions out of `^`) and which **end** it
(transitions into `^`). With **Laplace +1 smoothing** (add one to every count so
nothing is impossible):

```
count[a,b] = 1 + #{ adjacent (a→b) pairs in the framed corpus }
T[a,b]     = count[a,b] / Σ_c count[a,c]
```

We store the **integer counts**, not the normalised probabilities — see §5 for
why (determinism).

### 2.3 Sampling = inverse-CDF over integer weights

To draw the next symbol given the current symbol `a`, draw a uniform integer
`x ∈ [0, rowTotal(a))` and walk the row accumulating counts; return the symbol
whose cumulative band contains `x` ("roulette-wheel" sampling). Because the
weights are integers and the draw is an integer modulo, the choice is exact and
identical on any hardware.

### 2.4 The objective (scoring)

A real reward is `R(s) = QED(s)`, a docking score, etc. Here `R` is a **toy
integer drug-likeness proxy** (`score_molecule` in `generator.h`): it rewards
polar-atom content and the presence of a ring, penalises unbalanced branches,
and prefers a drug-like size window. It returns milli-reward (fixed-point
integer). The "best" molecule is `argmax_s R(s)` over the generated batch.

---

## 3. The algorithm

```
load corpus (training SMILES, n_generate, seed)
train_model:                                  # O(total training chars)
    count[a,b] = 1 for all a,b                # Laplace smoothing
    for each training string s:
        prev = ^                              # sentinel start
        for each char c in s (in alphabet):
            count[prev, sym(c)] += 1
            prev = sym(c)
        count[prev, ^] += 1                   # sentinel end
    rowTotal[a] = Σ_b count[a,b]

generate_and_score (n_generate independent jobs):    # the parallel part
    for i in 0 .. n_generate-1:
        rng   = seed_stream(seed, i)          # reproducible per-molecule stream
        s     = ""                            # build the molecule
        cur   = ^
        repeat up to MAX_LEN:
            nxt = sample_next(count, rowTotal, cur, rng)   # inverse-CDF draw
            if nxt == ^: break                # model chose to stop
            append char(nxt) to s; cur = nxt
        score[i]  = R(s);  length[i] = |s|

report: fraction of hits, mean reward, argmax molecule
```

**Complexity.** Training is `O(C)` in the total number of training characters
(done once on the host). Generation is `O(n_generate · L · K)`: each of
`n_generate` molecules takes up to `L = MAX_LEN` steps, each step an `O(K)`
inverse-CDF walk. Serially that is one big loop; in parallel the `n_generate`
molecules are fully independent, so the wall-clock time is `O(L · K)` per thread
with `n_generate` threads in flight.

---

## 4. The GPU mapping

This is the **per-thread RNG / "Monte-Carlo histories"** pattern
(PATTERNS.md §1; the same idiom as flagship 5.01 Monte-Carlo dose).

- **Thread-to-data mapping.** One thread generates and scores **one** molecule:
  `i = blockIdx.x * blockDim.x + threadIdx.x` owns molecule `i`, guarded by
  `if (i >= n_generate) return;` for the ragged last block. Block size 256 is a
  good occupancy default on sm_75–sm_89.
- **The model lives in `__constant__` memory.** Every thread reads the same
  read-only `K×K` transition table (~784 bytes) and never writes it. Constant
  memory's broadcast cache serves one fetched value to a whole warp — ideal for
  a read-shared parameter (the same trick as the query fingerprint in flagship
  1.12). The host uploads it once with `cudaMemcpyToSymbol`.
- **No atomics, no shared memory, no synchronisation.** Each thread writes only
  its own `score[i]` and `length[i]` (coalesced, since adjacent threads write
  adjacent slots). Molecules never interact — embarrassingly parallel.
- **Per-thread state in registers/local memory.** The molecule string is built
  in a per-thread `char buf[MAX_LEN+1]` and the RNG state is one 64-bit register.
- **Reproducible per-index seeding** (`rng_seed(seed, i)`) is what lets molecule
  `i` be **bit-identical** to the CPU's molecule `i`, which is the whole basis of
  exact verification.

### Memory-hierarchy summary

| Data | Space | Why |
|---|---|---|
| transition counts + row totals | **constant** | read by all threads, never written; broadcast cache |
| RNG state, loop indices | **registers** | tiny, per-thread, hot |
| molecule character buffer | **local** (per-thread) | `MAX_LEN+1` bytes scratch, no sharing |
| `score[]`, `length[]` outputs | **global** | one coalesced write per thread |

---

## 5. Numerical considerations

- **Integers, not floats — for determinism.** A floating-point roulette wheel
  (draw a float, compare against normalised probabilities) would risk the CPU and
  GPU rounding a borderline cumulative sum differently and picking a different
  token, diverging the whole molecule. We keep the transition weights as
  **integers** and sample with an **integer modulo + integer cumulative sum**, so
  the sampled token is exact and identical everywhere (PATTERNS.md §3). The
  reward is likewise an **integer** (milli-reward), so the mean-reward sum is
  order-independent (we accumulate in a 64-bit `long long` to avoid overflow at
  large `n_generate`).
- **Shared RNG.** Both sides use the same `splitmix64` counter-based generator
  (a few integer ops, identical on host and device). Production GPU code would
  use cuRAND, but cuRAND would *not* match a host RNG bit-for-bit, breaking exact
  verification — so we deliberately share a simple reproducible RNG instead.
- **Warp divergence.** Molecules have different lengths, so threads in a warp run
  their generation loops for different numbers of steps; the warp waits for its
  slowest thread. This is the classic MC-on-GPU cost. With thousands of molecules
  per launch it is hidden by having many warps resident; production samplers also
  batch by length. (No correctness impact — only throughput.)
- **No `--use_fast_math`.** There is no floating-point hot path to speed up; the
  arithmetic is integer, so fast-math is irrelevant and left off.

---

## 6. How we verify correctness

Two layers:

1. **Exact GPU == CPU (tolerance literally zero).** The CPU reference
   (`reference_cpu.cpp`) and the GPU kernel (`kernels.cu`) call the **identical**
   shared functions in `generator.h` (`rng_seed`, `generate_molecule`,
   `score_molecule`) with the **identical** per-index seed. So molecule `i` must
   be bit-identical on both, and `main.cu` checks every `(score[i], length[i])`
   pair for **exact integer equality** — `mismatches` must be 0. This is the
   strongest possible check (PATTERNS.md §4 "exact").
2. **Cross-checking the reduction.** The "best molecule" index is found two ways:
   the CPU reference tracks its own argmax during generation, and `main.cu`
   computes the argmax from the **GPU** scores. The stderr line asserts they
   agree, validating that the deterministic tie-break (highest score, lowest
   index on ties) is consistent across paths.

Edge cases handled: empty/comment lines and a malformed header (loud
`runtime_error`); the ragged last thread block (in-kernel guard); a row total
that is always `≥ K > 0` thanks to Laplace smoothing (no divide-by-zero in
sampling); molecule length capped at `MAX_LEN`.

---

## 7. Where this sits in the real world

This toy maps cleanly onto the production stack — every simplified piece has a
deep-learning analogue:

| This project (teaching) | Production de-novo design |
|---|---|
| First-order Markov `P(s_t \| s_{t-1})` | RNN/transformer LM `P(s_t \| s_1…s_{t-1})` (REINVENT4); diffusion over 3-D graphs (DiffSBDD, TargetDiff) |
| "Training" = counting transitions | Gradient descent on millions of SMILES, cuDNN-accelerated, FP16 mixed precision |
| Constant-memory transition table | Learned network weights in global/tensor memory |
| Inverse-CDF sampling of one token | A full forward pass + softmax + sampling per token |
| Toy integer reward (`score_molecule`) | QED, SA score, docking (AutoDock-GPU / DiffDock), ADMET predictors |
| `argmax` over a batch | REINFORCE / PPO updates that *re-weight the model* toward high-reward regions; curriculum learning (REINVENT4) |
| Per-thread independent generation | GPU-batched RL rollouts — thousands of candidate molecules per GPU-second |

What the toy is honest about **not** doing: it does not guarantee chemically
valid SMILES (a real model learns validity from data and is checked with RDKit);
its first-order memory cannot capture long-range constraints like matching ring
closures or branch nesting (a transformer's attention can); and its scorer is a
made-up proxy, not a validated medicinal-chemistry objective. The **GPU pattern**
it teaches — independent per-thread stochastic generation from a shared
read-only model, scored in parallel, verified exactly against a CPU twin — is
exactly the rollout pattern that real RL fine-tuning runs at scale.

**Further reading:** REINVENT4 (SMILES RL, Apache-2.0), DiffSBDD / TargetDiff
(3-D structure-based diffusion), DiffDock (diffusion pose generation), DeepChem
(broad ML drug-discovery toolkit), and the MOSES / GuacaMol benchmarks for how
the field measures distribution-learning and goal-directed quality.

> **Not for clinical use.** Everything here is educational and synthetic; no
> output is a real or usable molecule.
