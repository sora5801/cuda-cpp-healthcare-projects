# THEORY — 2.14 Protein-Ligand Co-Folding

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

This project is a **reduced-scope teaching version** (CLAUDE.md §13). It keeps the
real architecture — a reverse-diffusion loop whose every step is a self-attention
pass over a joint protein+ligand token sequence — but replaces the *trained*
score network with a *fixed analytic* one. That choice makes every line of math
inspectable and makes the GPU and CPU agree to machine precision. §7 describes the
full learned model.

---

## 1. The science

**The problem.** A drug molecule (ligand) works by binding to a pocket on a
target protein. To design or rank drugs *in silico* we need the **3-D structure
of the complex**: where the protein's atoms sit, and how the ligand is posed
inside the pocket. Classically this was two steps: (1) determine/predict the
protein structure, then (2) *dock* the ligand into the (frozen) pocket.

**Co-folding** does both at once. Given a protein sequence and a ligand, a single
generative model predicts the protein conformation **and** the ligand pose
**jointly**, letting the pocket adjust to the ligand and vice versa ("induced
fit"). This is what Boltz-1 and AlphaFold3 do, reaching pose accuracy that
rivals physics-based free-energy methods in minutes per complex.

**Why "diffusion".** These models are **denoising diffusion** generators. Imagine
taking the true complex and gradually adding random noise to every atom's
position until it is an unstructured cloud (the *forward* process). A diffusion
model learns to **reverse** that: starting from noise, it repeatedly nudges atoms
toward plausible positions until a clean structure emerges. Each nudge is computed
by a neural network that looks at the whole complex — and "looking at the whole
complex" is **attention**.

**What we model here.** A toy complex of `N` tokens — protein backbone (Cα)
tokens forming a binding-pocket rim, plus ligand heavy-atom tokens inside it — and
the reverse-diffusion loop that folds a noise cloud back into the planted pose.

## 2. The math

**State.** Each token `i` has a 3-D position `xᵢ ∈ ℝ³` (what we denoise) and a
fixed type `tᵢ ∈ {protein, ligand}`. We also keep its **native** position
`x*ᵢ` (the planted answer) and define a noised start `x_T,ᵢ = x*ᵢ + σ·εᵢ`,
`εᵢ ~ 𝒩(0, I₃)`, `σ =` `noise_scale`.

**Attention.** For a query token `i` and key token `j`, the unnormalized score is
**geometric (radial-basis) attention** plus a same-type bonus:

```
logit(i,j) = −‖xᵢ − xⱼ‖² / (2·temp²)  +  type_bias · 1[tᵢ = tⱼ]
```

The first term is a Gaussian distance kernel: it is `0` when the two tokens
coincide and decays with separation, so a query attends most to keys near its
**current** position. `temp` is the bandwidth ("temperature"): small = sharp
(near one-hot on the local neighbourhood), large = diffuse averaging. Softmax over
`j` gives weights:

```
a_ij = exp(logit(i,j) − m_i) / Σ_k exp(logit(i,k) − m_i),   m_i = max_k logit(i,k)
```

The `m_i` subtraction is the standard numerically-stable softmax (prevents
`exp` overflow); it cancels in the ratio.

> **Why this *is* attention.** Expand the square:
> `−‖q−k‖² = 2 q·k − ‖q‖² − ‖k‖²`. The `2 q·k` term is exactly scaled
> dot-product attention on the position features; `−‖q‖²` is constant across keys
> `j` and cancels in the softmax; `−‖k‖²` is the usual key-norm correction. RBF
> attention is the dot-product attention you get from coordinate features — we use
> the RBF form because it makes "attend to your neighbourhood" explicit.

**Predicted clean position (the score's target).** Aggregate the keys' **native**
targets with the attention weights:

```
x̂0ᵢ = Σ_j a_ij · x*ⱼ
```

i.e. "attention points token `i` to where its trusted neighbours belong." The
diffusion **score** (∇ log-density) then points from the noisy `xᵢ` toward `x̂0ᵢ`.

**DDIM update.** The deterministic (DDIM-style) sampler moves a fixed fraction
`α =` `step_frac ∈ (0,1]` of the way each step:

```
xᵢ ← xᵢ + α·(x̂0ᵢ − xᵢ)
```

Iterating `T` times is a contraction toward the attention-consistent fixed point,
which (by construction of `x̂0`) is the native complex `x*`. **Objective / success
metric:** root-mean-square deviation to native,
`RMSD = sqrt( (1/N) Σᵢ ‖xᵢ − x*ᵢ‖² )`, should fall from `σ·√3`-ish toward `0`.

## 3. The algorithm

```
build x_T = x* + σ·ε                       # forward-diffusion endpoint (noised)
for step = 1 … T:                          # reverse diffusion
    for each query token i (in parallel):  # ONE attention pass
        m  = max_j logit(i,j)              # pass 1: stable-softmax max
        Z  = Σ_j exp(logit(i,j) − m)       # pass 2: softmax denominator
        x̂0 = (Σ_j exp(logit(i,j) − m)·x*_j) / Z
        x_next[i] = x[i] + α·(x̂0 − x[i])   # DDIM blend
    swap(x, x_next)                        # double-buffer (no in-step races)
report RMSD(x, x*) and the ligand pose
```

**Complexity.** Each step is all-pairs: `O(N²)` logit evaluations, each `O(1)`
(a 3-vector difference). A clean-position aggregation is `O(N²·3)`. Over `T`
steps the serial cost is `Θ(T·N²)`. The **parallel work** is the same `Θ(T·N²)`,
but the **depth** per step is `O(log N)` (the reduction tree) instead of `O(N)`:
the `N` query tokens are independent, and within a query the key loop reduces in
parallel. Arithmetic intensity is modest (a few flops per key load), so at small
`N` the kernel is **launch/bandwidth-bound** — the honest-timing caveat (§5, §7).

## 4. The GPU mapping

**Decomposition: one block per query token.** Token `i = blockIdx.x` is updated by
one block of `THREADS_PER_TOKEN = 64` threads (two warps — enough to hide the
strided global loads). The threads split the key loop `j = tid, tid+nt, …` and
combine partials in shared memory. This is the **FlashAttention** shape: parallel
over the key dimension, `O(1)` extra storage per query, an online (two-pass)
softmax.

```
grid  = N blocks                      (one per query token i)
block = 64 threads                    (cooperate over all N keys)
shmem = 64 doubles                    (reused for the max, then 4 sum reductions)

   query token i  ────────────────────────────────────────────►  x_next[i]
        │  block i
        ▼
   ┌───────────── threads 0..63 (each takes a strided slice of keys) ─────────────┐
   │  PASS 1:  local max logit  ──► block_reduce_max ──►  m_i (broadcast via s[0]) │
   │  PASS 2:  local Σ exp, Σ exp·x*  ──► block_reduce_sum ×4 ──►  Z, (Sx,Sy,Sz)   │
   │  thread 0:  x̂0 = (Sx,Sy,Sz)/Z ;  x_next[i] = ddim_blend(x[i], x̂0, α)         │
   └─────────────────────────────────────────────────────────────────────────────┘
```

**Memory hierarchy.**
- **Global**: `pos`, `target`, `types` (read), `pos_next` (write). Each block
  owns a distinct output token, so there are **no atomics and no cross-block write
  conflicts** (contrast the k-means flagship 11.09, which *does* need atomics).
- **Shared**: one `blockDim`-length scratch array, reused sequentially for the max
  reduction and then the four sum reductions (`Z`, `Σx`, `Σy`, `Σz`) — a common
  space-saving idiom that keeps shared memory to a single array.
- **Registers**: the query position/type and each thread's partials.

**Host loop & ping-pong.** The host runs the `T`-step loop, launching the kernel
once per step and swapping two device position buffers (read the frozen state,
write the next, swap) — the same double-buffer discipline as the reaction-diffusion
stencil (project 14.02), here with attention instead of a Laplacian.

**No CUDA library is linked** beyond the runtime. We hand-roll the softmax and the
reduction precisely so the learner sees them. In production this step is a call
into **FlashAttention2 / cuDNN**; writing it by hand would mean (and here does
mean) implementing the tiled online-softmax that those libraries optimize to the
memory hierarchy. See PATTERNS.md §5 for the "use the library, but explain it" rule.

## 5. Numerical considerations

**Precision: FP64.** We use `double` throughout. This is a *teaching* choice: it
shrinks the CPU-vs-GPU gap so the focus stays on the algorithm. Real co-folding
runs in **FP16/BF16** for throughput and accepts the accuracy hit (the learned
model is robust to it); we trade that speed for legibility.

**Determinism & the reduction-order caveat.** The per-token math is the shared
`denoise_token`/`attention_logit` in `cofold.h`, called identically by CPU and
GPU — so in exact arithmetic they are bit-identical. But floating-point addition
is **not associative**, and the GPU's block reduction sums partials in a **tree**
order while the CPU sums **left-to-right**. Over `T = 160` steps these orders
diverge by ~`1e-13`/step (also fed by fused-multiply-add contraction differences),
accumulating to ~`1e-4`. In this demo the measured worst position difference is
~`9e-16` (the system is small and well-conditioned), but the *tolerance* is set
for the general case. **No atomics** are used, so there is no nondeterminism from
atomic ordering; stdout is byte-identical every run (timings go to stderr —
PATTERNS.md §3).

**Geometric-attention pitfall (worth knowing).** Because `x̂0` is a convex
combination of native targets, a token can only reach its *own* native if its
attention is near one-hot on itself. Tokens that are mutually closest and
symmetric (e.g. tightly-clustered ligand atoms) instead average toward their
**centroid** — a real failure mode of distance-kernel attention. We avoid it by
spacing the synthetic ligand atoms ~1.8 Å apart so each has a distinct
neighbourhood (see `scripts/make_synthetic.py`); shrink the spacing and you can
watch two atoms collapse onto one. A learned model dodges this with expressive,
non-geometric features.

## 6. How we verify correctness

Two independent checks (PATTERNS.md §4):

1. **GPU vs CPU agreement.** `src/reference_cpu.cpp` runs the *same* reverse
   diffusion serially (calling the same `cofold.h` math). `main.cu` compares the
   final positions element-wise; the worst absolute difference must be
   `≤ 1e-3`. We pick `1e-3` (not bit-exactness) honestly: it is the physically
   negligible band that covers the reduction-order/FMA drift of a long iterative
   double-precision loop (§5), while being ~`10¹²` times smaller than the
   coordinates (`O(10) Å`). Agreement between an *independent serial*
   implementation and the parallel GPU one is strong evidence both are correct.
2. **Science-level recovery (analytic check).** The reverse diffusion should fold
   the noise cloud back to the planted complex, so the final **RMSD-to-native**
   must drop below `0.5 Å`. The demo reports `start=1.5503 → final=0.0120`, and
   each ligand atom lands on its own native coordinate — confirming the *pose*,
   not just CPU==GPU. (A real pose-prediction tool reports ligand-RMSD to the
   crystal pose, with `< 2 Å` counted "correct".)

**Edge cases handled:** ragged last block (grid sized to `N`, `i ≥ N` guarded),
empty union/zero denominator guarded (`inv = 0`), and a malformed sample file
throws with a clear message instead of producing garbage.

## 7. Where this sits in the real world

This teaching version differs from production co-folding in exactly one place —
**the score** — and in three details:

- **Learned vs analytic score.** Boltz-1 / AlphaFold3 *learn* `x̂0` (or the noise
  `ε`) with a deep transformer trained on the PDB: triangle/pair attention over
  residue and atom representations, MSA features, template features, and a
  diffusion module. Our `x̂0` is a fixed geometric average of *known* native
  targets — so it cannot generalize to an unseen complex; it only re-folds the
  one it was given. The **loop structure** (denoise `T` times, attention each
  step) is faithful; the **network** is the stub.
- **Tokens & features.** Real models use rich learned embeddings (residue type,
  atom element, bond graph, chirality) and **cross-attention** between protein and
  ligand streams; we use raw coordinates + a type flag and let the shared sequence
  do the cross-attention implicitly.
- **Sampler & confidence.** Production uses the stochastic DDPM (or DDIM with a
  trained noise schedule) and emits **pLDDT/iPAE confidence**; we use a
  deterministic DDIM (for reproducible stdout) and report RMSD only because we
  planted the native.
- **Hardware.** FlashAttention2 + cuDNN in FP16/BF16, multi-GPU model parallelism
  for large complexes; here FP64 + a hand-rolled kernel for clarity.

To go further: study Boltz-1's diffusion sampler and joint-token representation,
then DiffDock to contrast diffusion *docking* (fixed protein) with *co-folding*.

---

## References

- **AlphaFold3** — Abramson et al., *Nature* (2024); the diffusion-module +
  confidence-head architecture this project caricatures.
  https://github.com/google-deepmind/alphafold3
- **Boltz-1** — open, GPU co-folding of protein–ligand–nucleic-acid complexes;
  the most readable production diffusion sampler to study.
  https://github.com/jwohlwend/boltz
- **NeuralPLexer3** — state-specific co-folding with CUDA; how multiple
  conformational states are sampled. https://github.com/zrqiao/NeuralPLexer
- **DiffDock** — Corso et al. (2023); diffusion *docking* into a fixed protein, a
  useful contrast. https://github.com/gcorso/DiffDock
- **DDPM / DDIM** — Ho et al. (2020), *Denoising Diffusion Probabilistic Models*;
  Song et al. (2021), *Denoising Diffusion Implicit Models* — the sampler math.
- **FlashAttention2** — Dao (2023) — the tiled online-softmax kernel our toy
  block-attention imitates.
- **PoseBusters** — Buttenschoen et al. (2024) — the physically-valid pose
  benchmark co-folding models are judged on. https://github.com/maabuu/posebusters
