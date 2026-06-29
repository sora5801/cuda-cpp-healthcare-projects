# THEORY — 1.5 Free Energy Perturbation / Thermodynamic Integration

> A deep dive for a reader who knows C++ but is new to CUDA *and* new to free-energy
> methods. This project is a deliberately **reduced-scope teaching model**: the GPU
> pattern and the statistical mechanics are real; the *system* is a 1-D toy whose
> answer is known in closed form, so we can verify the method exactly. See the final
> section for how production FEP differs.
>
> _Educational only — not for clinical use._

---

## The science

A central question in drug discovery is: *does ligand B bind a protein more tightly
than ligand A?* The thermodynamically rigorous answer is the **binding free-energy
difference** ΔΔG. Free energy — not just potential energy — is what governs
equilibrium: it folds in **entropy** (how many microstates a configuration has) as
well as enthalpy. You cannot read ΔG off a single structure; it is an *ensemble*
property, an average over all the thermally accessible wiggles of the system.

Two molecules A and B are usually too different to compare by brute force. The trick
(Kirkwood, 1935) is **alchemy**: build a fictitious, continuous path parameterised by
a coupling variable **λ ∈ [0, 1]** that smoothly *morphs* the Hamiltonian of state A
(λ=0) into that of state B (λ=1). The path need not be physical — atoms can fade in
and out — because free energy is a **state function**: ΔG depends only on the
endpoints, not the route. We then either

- **FEP** (Free Energy Perturbation, Zwanzig 1954): estimate ΔG from the *exponential
  average* of energy differences between neighbouring λ-states, or
- **TI** (Thermodynamic Integration): integrate the *average force along λ* over the
  path.

This project implements **TI** on a model where the two "ligands" are two harmonic
springs of different stiffness — capturing the essential machinery (the λ-path, the
ensemble averages, the integration) while staying simple enough to check against a
formula.

---

## The math

### The model system

One particle at position `x ∈ ℝ`. Two end states are harmonic wells:

$$U_A(x) = \tfrac12 k_A (x - x_{0A})^2, \qquad U_B(x) = \tfrac12 k_B (x - x_{0B})^2.$$

We couple them **linearly** (the simplest λ-schedule):

$$U(x, \lambda) = (1-\lambda)\,U_A(x) + \lambda\,U_B(x).$$

Note $U(x,\lambda)$ is itself harmonic, with effective stiffness $k(\lambda) =
(1-\lambda)k_A + \lambda k_B$ and a λ-dependent centre — a fact we exploit only for
intuition, not in the code (the sampler never assumes it).

### Free energy and the TI identity

At temperature $T$, with $\beta = 1/k_BT$ (we set the Boltzmann constant $k_B=1$, so
$\beta = 1/kT$), the free energy of a state with potential $U_\lambda$ is

$$G(\lambda) = -kT \ln Z(\lambda), \qquad Z(\lambda) = \int e^{-\beta U(x,\lambda)}\,dx.$$

Differentiate $G$ with respect to λ (the $\ln Z$ derivative turns into a Boltzmann
average):

$$\boxed{\;\frac{dG}{d\lambda} = \left\langle \frac{\partial U}{\partial \lambda} \right\rangle_\lambda\;}
\quad\Longrightarrow\quad
\Delta G = G_B - G_A = \int_0^1 \left\langle \frac{\partial U}{\partial \lambda} \right\rangle_\lambda d\lambda.$$

Here $\langle\cdot\rangle_\lambda$ is the equilibrium (Boltzmann) average **in the
coupled ensemble at coupling λ**. For our linear coupling,

$$\frac{\partial U}{\partial \lambda} = U_B(x) - U_A(x),$$

independent of λ *for a fixed x* — so the entire λ-dependence of the TI integrand
comes from how the **sampling distribution** shifts as λ changes. That is the crux of
TI and the thing the Monte Carlo must capture.

### The closed-form answer (our ground truth)

A 1-D harmonic oscillator has an analytic partition function
$Z(k) = \sqrt{2\pi kT / k}$, hence $G(k) = \tfrac12 kT \ln\!\big(k/(2\pi kT)\big)$ up
to an additive constant. The constant and the well centres $x_0$ **cancel** in a
difference, leaving

$$\boxed{\;\Delta G_{\text{analytic}} = \tfrac12\,kT \,\ln\!\frac{k_B}{k_A}\;}$$

For the committed sample ($k_A=1,\,k_B=4,\,kT=1$): $\Delta G = \tfrac12\ln 4 = \ln 2
\approx 0.693147$. The TI estimate must converge to this. (A lovely teaching point:
shifting a spring's centre changes its energy landscape but **not** its free energy.)

---

## The algorithm

We discretise the λ-integral into **W windows** at $\lambda_w = w/(W-1)$, estimate
each $\langle \partial U/\partial\lambda\rangle_{\lambda_w}$ by Monte Carlo, then
integrate with the trapezoid rule.

**Per-window Metropolis Monte Carlo** (samples the Boltzmann distribution
$\propto e^{-\beta U(x,\lambda_w)}$):

```
x ← x_init ; Ux ← U(x, λ_w)
repeat (equil + samples) times, step n:
    x' ← x + (2·u₀ − 1)·step          # u₀ ∈ [0,1) uniform, symmetric proposal
    dE ← U(x', λ_w) − Ux
    if dE ≤ 0 or u₁ < exp(−dE/kT):    # Metropolis acceptance, u₁ ∈ [0,1)
        x ← x' ; Ux ← U(x', λ_w)
    if n ≥ equil:                     # after burn-in
        sum ← sum + (U_B(x) − U_A(x)) # accumulate ∂U/∂λ
⟨∂U/∂λ⟩ ← sum / samples
```

**TI integration** (composite trapezoid, spacing $h = 1/(W-1)$):

$$\Delta G_{TI} = h\left(\tfrac12 f_0 + f_1 + \dots + f_{W-2} + \tfrac12 f_{W-1}\right),
\quad f_w = \langle\partial U/\partial\lambda\rangle_{\lambda_w}.$$

### Complexity

Let $W$ = windows and $S$ = `equil + samples` steps per window.

- **Serial (CPU):** $O(W \cdot S)$ — one chain after another.
- **Parallel (GPU):** the $W$ chains are independent, so with $W$ threads the
  *wall-clock* work is $O(S)$ (plus launch overhead) — a $W\times$ reduction in the
  ideal case. Real FEP pushes $W$ to dozens and $S$ to millions, and replaces each
  thread's 1-D move with a full MD step over thousands of atoms; the *mapping* is the
  same. (Production codes run one whole GPU **per** window — the windows are the
  coarse parallel axis; the atoms are the fine one.)

---

## The GPU mapping

This is the **ensemble-of-independent-jobs** pattern (`docs/PATTERNS.md` row "same
sampler for many parameter sets"; exemplars `9.02` SEIR, `13.02` PBPK).

```
            λ-windows  (one independent MC chain each)
   w =  0      1      2            ...            W-1
       ┌──┐  ┌──┐  ┌──┐                          ┌──┐
thread │t0│  │t1│  │t2│   one GPU thread per →   │tN│
       └──┘  └──┘  └──┘    window, full chain    └──┘
        │     │     │      in registers           │
        ▼     ▼     ▼                              ▼
      dvals[0] dvals[1] ...                    dvals[W-1]   → trapezoid → ΔG
```

- **Thread-to-data map:** `w = blockIdx.x*blockDim.x + threadIdx.x`; thread `w` owns
  window `w`, runs its entire chain in registers/local memory, and writes one
  `double` to `dvals[w]`. No shared memory, **no atomics**, no inter-thread comms —
  the cleanest possible parallel decomposition (`kernels.cu`).
- **Memory hierarchy:** the `AlchemyConfig` is passed **by value** into the kernel, so
  it rides in the kernel's *parameter/constant* space — read identically by every
  thread, never written. The only global-memory traffic is the final scatter of $W$
  results, which is negligible. The chain state (`x`, `Ux`, `sum`) lives in
  **registers**.
- **Block size:** 128. The per-thread work is a long serial loop (compute-bound,
  branch-y), not memory-bound, so we do not need a huge block to hide latency; 128
  keeps register pressure modest. Occupancy is *not* the bottleneck here — a real
  run has tens of windows, so this kernel is **launch-bound** on tiny inputs (see
  "honest timing" below).
- **Shared `__host__ __device__` core (`alchemy.h`):** the potentials, the RNG, and
  `run_chain()` are compiled **once** and run on both sides. The CPU reference loops
  over windows; the GPU runs one thread per window — calling the *same* function.
  This is what makes verification essentially exact (next section).

---

## Numerical considerations

### Precision

All sampling is in **FP64 (double)**. Free-energy differences are small numbers
extracted from large fluctuating energies, so single precision would accumulate
visible drift over $10^4$–$10^5$ steps. `sm_75`+ run FP64 at reduced throughput, but
correctness/determinism dominate this teaching demo, not speed.

### Determinism — the counter-based RNG

Monte Carlo needs randomness, yet the demo's stdout must be **byte-identical every
run** (`docs/PATTERNS.md §3`). We therefore use a **stateless, counter-based RNG**:
the $n$-th uniform of window $w$ on channel $c$ is a *pure hash* of the integer key
built from $(w, n, c)$, mixed with the SplitMix64 finalizer and scaled by $2^{-53}$:

$$u = \text{splitmix64}\big((w \ll 40)\oplus(n\ll 4)\oplus c\big)\big/2^{53}.$$

There is no mutable seed and no per-thread state array, so the random stream does
**not** depend on thread scheduling. The CPU and GPU hash the *same* keys → draw the
*same* numbers → produce the *same* chain. (This is exactly why production GPU-MC
codes favour counter-based generators such as NVIDIA cuRAND's Philox.) Two channels
(`c=0` proposal, `c=1` accept-test) keep the two draws a step needs from colliding.

> **A bug worth remembering.** The first draft scaled the hash by the wrong constant
> (≈ 2× too large), which clamped every uniform to $[0, 0.5)$. The chain then drifted
> off to $x \approx -31$ and TI returned ≈ 1346 instead of 0.693. The fix was a
> one-line constant ($2^{53} = 9007199254740992$). Lesson: a biased RNG fails
> *silently* — always sanity-check $\langle u\rangle \approx 0.5$.

### Why the GPU and CPU agree to ~1e-13, not exactly 0

Same code + same RNG would suggest *bit-identical* results, and we very nearly get
them (worst per-window difference ≈ $8\times10^{-14}$). The residual comes from the
one place the two compilers may differ: the **`exp()`** in the Metropolis test and
**FMA contraction** in the potential, which can disagree by 1–2 ulp. Over a long
chain a *single* differing accept/reject decision can nudge the average, so we verify
to `1e-9` and call it "essentially exact" (`docs/PATTERNS.md §4`).

### Statistical + discretisation error in ΔG

The TI estimate is a *statistical* quantity, so it only **approaches** the analytic
ΔG. Two error sources:

1. **Finite sampling** — each $\langle\partial U/\partial\lambda\rangle$ has Monte
   Carlo noise $\sim 1/\sqrt{\text{samples}}$.
2. **Trapezoid discretisation** — the TI integrand is steeply curved near $\lambda=0$
   (see the printed curve: $3.64 \to 2.17 \to 1.21\dots$), so a coarse 11-window grid
   slightly *over*-estimates the integral. That is why the demo gives $0.711$ vs the
   true $0.693$ — a real, visible bias, not a bug. More windows shrink it
   (`make_synthetic.py --windows 41`).

We verify ΔG against the analytic answer to a **physical tolerance of `5e-2`** and
*print the gap* so the learner sees the convergence story honestly.

---

## How we verify correctness

Two independent checks, both in `main.cu`:

1. **GPU vs CPU reference** (`reference_cpu.cpp`): the serial loop and the parallel
   kernel call the same `run_chain()`; worst per-window difference must be `≤ 1e-9`.
   This catches *implementation* bugs (indexing, memory, launch config).
2. **TI vs analytic** $\tfrac12 kT\ln(k_B/k_A)$: the integrated ΔG must be within
   `5e-2` of the closed form. This catches *method/physics* bugs — it validates that
   the sampler actually draws the Boltzmann distribution and that TI is wired up
   correctly. (A passing CPU==GPU with wrong physics would still fail *here*.)

The committed sample embeds a known answer ($\ln 2$) so the second check is
meaningful (`docs/PATTERNS.md §6`).

---

## Where this sits in the real world

This toy keeps the *scaffolding* of FEP/TI and discards the *system*. Production
relative binding free-energy (RBFE) calculations differ in every dimension of size:

- **The system** is a solvated protein–ligand complex (10⁴–10⁵ atoms), not one
  particle. Each "MC/MD step" is a full force evaluation: bonded terms, Lennard-Jones,
  and **PME electrostatics** (see project `1.2`) — typically run as Langevin/Velocity-
  Verlet **MD**, not 1-D Metropolis.
- **Soft-core potentials** (Beutler/Zacharias) replace our plain linear coupling so
  that atoms appearing/disappearing near $\lambda=0$ or $1$ do not create singular
  $1/r$ forces ("endpoint catastrophe"). Our harmonic wells have no such singularity,
  which is *why* linear coupling suffices here.
- **Estimators:** real pipelines prefer **BAR/MBAR** (Bennett / multistate Bennett
  acceptance ratio, `pymbar`/`alchemlyb`) over raw TI — they use the *overlap* between
  neighbouring λ-ensembles to extract ΔG with lower variance, and report an overlap
  matrix to diagnose poorly-converged windows. **REST2** (replica exchange with solute
  tempering) further accelerates sampling.
- **Parallelism:** the windows are run as an embarrassingly parallel array, often
  **one GPU per λ-window**, with NCCL only for replica-exchange swaps — the same
  coarse decomposition this kernel models, scaled out across a cluster.
- **Tools:** OpenFE, GROMACS FEP (+`alchemlyb`), OpenMMTools, and AMBER
  `pmemd.cuda` TI. The **Merck/OpenFE benchmark** (8 targets, experimental ΔΔG) is the
  standard accuracy yardstick; well-tuned RBFE reaches ~1 kcal/mol RMSE, accurate
  enough to rank congeneric series in lead optimisation.

**Not for clinical or real decision-making use.** The data here is synthetic and the
model is pedagogical.
