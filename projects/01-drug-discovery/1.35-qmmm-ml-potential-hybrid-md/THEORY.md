# THEORY — 1.35 QMMM/ML Potential Hybrid MD

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use. **Reduced-scope teaching version**
> (CLAUDE.md §13): the NNP is a tiny network with fixed synthetic weights, not a
> trained QM-accurate model._

---

## 1. The science

Molecular dynamics (MD) simulates how atoms move by repeatedly computing the
**force** on every atom and stepping Newton's equations forward in time. The
forces come from a **potential energy surface** (PES), `E(positions)`. The whole
game is: *how do you get an accurate `E` cheaply?*

- **Quantum mechanics (QM)** — solve (an approximation of) the Schrödinger
  equation, e.g. density-functional theory (DFT). Accurate enough to model
  **bond breaking and forming** (chemistry!), but so expensive you can only
  afford a few hundred atoms for a few picoseconds.
- **Molecular mechanics (MM)** — a hand-tuned classical force field: springs for
  bonds, Lennard-Jones for van der Waals, Coulomb for electrostatics. Cheap
  enough for millions of atoms and microseconds, but it **cannot break bonds**
  (the bonded terms are fixed) — so it cannot do reactions.

**QM/MM** is the classic compromise: treat the small *reactive center* (an enzyme
active site, say) with QM, and the large *environment* (the rest of the protein +
water) with MM. The two are coupled at a boundary. This is how computational
enzymology has worked for decades (Warshel & Levitt, Nobel 2013).

**The frontier (this project's catalog entry):** replace the expensive QM region
with a **machine-learned potential (NNP)** — a neural network *trained on QM
data* that reproduces QM forces at a tiny fraction of the cost. Now the reactive
center is fast too, so you can run **microsecond reactive MD at QM accuracy**.
Modern NNPs (MACE, NequIP, ANI) are *equivariant graph neural networks*; tools
like OpenMM-ML drop them into an MM environment. The hard parts are (a) getting
**training data** that covers the reactive intermediates (active learning), and
(b) handling **electrostatics across the boundary**.

**What this teaching version captures.** The *structure* of that hybrid: an
ML-described region + an MM region, coupled across a link atom, integrated with a
proper MD integrator, run as an ensemble. We swap the trained equivariant GNN for
a small **Behler–Parrinello** network with fixed weights so the whole thing fits
in one readable file and runs deterministically.

## 2. The math

### 2.1 The hybrid energy (mechanical embedding)

Partition the atoms into an ML set `M_ml` (the reactive center + the link atom)
and an MM set `M_mm` (the environment). The total energy is

```
E_total = E_NNP(M_ml)  +  E_MM(M_mm-M_mm pairs)  +  E_couple(M_ml-M_mm pairs)
```

- `E_NNP` — the neural-network energy of the ML region.
- `E_MM` — the classical energy *within* the environment.
- `E_couple` — how the two regions feel each other. In **mechanical embedding**
  (the simplest scheme, what we use) the coupling is just the classical
  nonbonded interaction across the boundary. (In **electrostatic embedding** the
  MM point charges additionally enter the NNP/QM Hamiltonian — see §7.)

ML–ML pairs are governed by `E_NNP`, **not** by `E_MM` (the network already
describes them); that non-double-counting is the whole point of the partition.

### 2.2 The NNP energy (Behler–Parrinello high-dimensional NN potential)

The NNP energy is a **sum of per-atom energies**, `E_NNP = Σ_{i∈M_ml} ε_i`. Each
`ε_i` depends only on atom `i`'s local environment, encoded by a vector of
**radial symmetry functions** (`N_G` of them):

```
G_i[k] = Σ_{j≠i, r_ij<R_c}  exp(-η (r_ij - μ_k)^2) · f_c(r_ij)        k = 0..N_G-1
```

- `r_ij = |x_i - x_j|` — distance to neighbor `j`.
- `μ_k` — the k-th Gaussian center (a "probe distance"); `η` — its sharpness.
- `f_c(r)` — a smooth cutoff so neighbors enter/leave continuously:
  `f_c(r) = ½(cos(π r/R_c) + 1)` for `r<R_c`, else 0. (Continuous value **and**
  derivative ⇒ no force discontinuity.)

The descriptor `G_i` (length `N_G`) is the input to a small multilayer perceptron
(one hidden layer of `N_HID` units, `tanh` activation):

```
z_h = b1_h + Σ_k W1[h][k] · G_i[k]
a_h = tanh(z_h)
ε_i = b2 + Σ_h W2[h] · a_h
```

The weights `W1, b1, W2, b2` are what training fits. **Here they are fixed,
synthetic constants** (`src/nnpmm.h`) — the architecture is real, the parameters
are a labeled surrogate.

### 2.3 Forces (analytic differentiation = "autograd by hand")

Force on atom `i` is `F_i = -∂E_total/∂x_i`. For the MM/LJ terms this is the
textbook LJ derivative. For the NNP term we **chain-rule** through the network:

```
∂ε_i/∂x_i = Σ_k (∂ε_i/∂G_i[k]) · (∂G_i[k]/∂x_i)
```

The network part (backprop through one layer, `tanh'(z)=1-tanh²(z)`):

```
∂ε_i/∂G[k] = Σ_h W2[h] · (1 - a_h²) · W1[h][k]
```

The descriptor part (product rule on the Gaussian × cutoff):

```
∂G[k]/∂r = (-2η(r-μ_k))·exp(-η(r-μ_k)²)·f_c(r)  +  exp(-η(r-μ_k)²)·f_c'(r)
```

and `∂r_ij/∂x_i = +sign(x_i-x_j)`, `∂r_ij/∂x_j = -sign(x_i-x_j)` (1-D). A real NNP
gets this gradient from PyTorch autograd; we write it explicitly so **nothing is a
black box** (CLAUDE.md §6). Both are the *same derivative*.

### 2.4 The MM/Lennard-Jones term

```
V_LJ(r) = 4ε[(σ/r)^12 - (σ/r)^6],     dV_LJ/dr = (4ε/r)[-12(σ/r)^12 + 6(σ/r)^6]
```

repulsive `^12` wall + attractive `^6` tail, minimum near `r = 2^{1/6}σ`.

### 2.5 The integrator (velocity-Verlet)

With unit masses (`m=1`, so acceleration = force):

```
x(t+dt) = x(t) + v(t)·dt + ½·a(t)·dt²
a(t+dt) = F(x(t+dt))
v(t+dt) = v(t) + ½·(a(t) + a(t+dt))·dt
```

Velocity-Verlet is **symplectic** and time-reversible, so total energy stays
bounded over long runs (it does not drift secularly) — the property we report.

## 3. The algorithm

For each ensemble member `idx` (independent):

```
1.  build the chain: x[i] = spacing·i, v=0, a=0
2.  perturb the link atom: x[LINK] += perturbation(idx, M, amp)   (active-learning probe)
3.  a ← F(x)                                  (prime accelerations; also record E0)
4.  repeat `steps` times:  velocity-Verlet step  (recomputes F each step)
5.  reduce to a summary: final PE, final total energy, max|force|, |E_final - E0|
```

**Force evaluation `F(x)` cost.** Per call: the NNP loop is
`O(N_ml · N · (N_G + N_HID))` and the LJ loop is `O(N²)`. With `N=8` this is
trivial, but the *shape* is what a 3-D production code scales up (replacing the
`O(N²)` neighbor scan with a cutoff neighbor list ⇒ `O(N)`).

**Per-trajectory cost.** `steps` force evaluations ⇒ `O(steps · N²)`. **Total
serial cost** `O(M · steps · N²)`. The time loop is inherently sequential (step
`t+1` needs step `t`), but the **M members are independent** — that is the
parallel axis.

## 4. The GPU mapping

**Pattern: ensemble — one thread per trajectory** (docs/PATTERNS.md §1; same as
flagships 9.02 SEIR, 13.02 PBPK). Each thread runs the *entire* velocity-Verlet
loop for one member in registers/local memory and writes a single `TrajResult`.

- **Thread-to-data map:** `idx = blockIdx.x·blockDim.x + threadIdx.x` → ensemble
  member `idx`. A guard `if (idx >= M) return;` covers the ragged last block.
- **Launch config:** block = 128 threads (multiple of the 32-lane warp; small
  enough that the per-thread MD state — `3·N_ATOMS` doubles — does not blow up
  register pressure / occupancy). Grid = `ceil(M / 128)` blocks.
- **Memory hierarchy:**
  - *Registers / local memory* hold each thread's `x,v,a` arrays and the
    trajectory state. No global traffic during integration.
  - *Global memory* is touched **once**, at the end, to write `out[idx]` — so
    the kernel is **compute-bound**, not bandwidth-bound (unusual and pedagogically
    nice). The tiny `EnsembleConfig` is passed **by value** (copied into every
    thread's parameter space), so even the inputs avoid global reads.
  - *No shared memory, no atomics, no inter-thread communication* — pure
    embarrassing parallelism. (Contrast 11.09's atomic centroid reduction.)
- **Divergence:** all members run the same number of steps and the same
  branch-light force loops, so warps stay coherent; only the per-member
  perturbation value differs. That coherence is *why* the ensemble maps cleanly.
- **No CUDA library needed.** Everything is hand-written; we link only `cudart`.
  (The *full* method would use cuBLAS for the equivariant tensor contractions and
  PyTorch autograd for forces — see §7.)

```
                 ensemble of M independent trajectories
   member:   0     1     2     3            ...          M-1
             │     │     │     │                          │
   GPU       ▼     ▼     ▼     ▼                          ▼
   thread:  t0    t1    t2    t3    ...                  t_{M-1}
             │     │     │     │                          │
         [build chain + perturb link atom]                │
             │     │     │     │                          │
         [ velocity-Verlet × steps  (in registers) ]      │
             │     │     │     │                          │
             └──── write out[idx] (single global store) ──┘
   blocks of 128 threads;  grid = ceil(M / 128)
```

## 5. Numerical considerations

- **Precision: FP64 (double) throughout.** MD energy conservation is sensitive;
  `tanh`, `exp`, and the LJ `^12` term all benefit from double precision. We do
  not use `--use_fast_math` (it would break the CPU/GPU bit-comparison and reduce
  `exp`/`tanh` accuracy).
- **Determinism.** No RNG anywhere — each member's perturbation is a fixed
  function of `idx` (`member_perturbation`), so the inputs are identical on CPU
  and GPU and across runs. There is **no parallel reduction over threads** (each
  thread owns one trajectory and writes one slot), so there is no atomic-ordering
  nondeterminism. stdout is therefore byte-identical every run (verified).
- **Why results still differ slightly (CPU vs GPU).** The GPU contracts
  `a*b+c` into a single fused multiply-add (FMA) with one rounding; the host
  compiler may use two roundings. That ~1 ulp per-op difference compounds over
  300 steps to `~4e-13` here — real, expected, and far below any meaningful
  scale. (docs/PATTERNS.md §4, the "long iterative solver" case.)
- **Stiffness / stability.** A configuration shoved into the LJ repulsive wall
  (member `m0`) has large forces; with a fixed `dt` its energy is conserved
  worse than near-equilibrium members. Velocity-Verlet stays *bounded* (it does
  not explode) but the bound grows with stiffness — the lesson being **stiff
  regions need a smaller `dt`** (Exercise 1).

## 6. How we verify correctness

Two independent checks:

1. **CPU vs GPU agreement.** `src/reference_cpu.cpp` integrates every trajectory
   serially; the GPU does it in parallel. Because *both call the same
   `__host__ __device__` functions in `nnpmm.h`*, they execute identical math, so
   agreement is strong evidence the parallelization is correct (not just that two
   buggy implementations agree). We compare all summary fields per member and
   take the worst absolute difference.
   - **Tolerance `1e-6`**, chosen to sit well above the observed FMA drift
     (`~4e-13`) yet far below any physically meaningful energy difference. Honest
     and documented (not pretended to be bit-exact).
2. **Physical sanity — energy conservation.** We report
   `max_idx |E_final(idx) - E_initial(idx)|`. A *symplectic* integrator should
   keep this small and bounded. The near-equilibrium members conserve energy to
   `~1e-3`; the stiffest member (`m0`) to `~0.2`. This validates the *dynamics*,
   not just CPU==GPU agreement (docs/PATTERNS.md §4, the "stronger check").

**Edge cases handled:** coincident atoms (`r<1e-9` guarded), neighbors outside
the cutoff (skipped), the ragged last GPU block (guarded), and a missing/garbled
config file (the loader throws).

## 7. Where this sits in the real world

What a production NNP/MM hybrid (MACE/NequIP + OpenMM-ML) adds that this teaching
version omits:

- **Equivariant message-passing networks.** Real NNPs are graph neural networks
  whose features transform correctly under rotation (E(3)-equivariance), built
  from spherical harmonics and tensor products (cuBLAS-heavy). Our scalar
  Behler–Parrinello descriptor is the *historical first generation* of the same
  idea; equivariant models are far more data-efficient and accurate.
- **Trained on QM data.** The weights come from fitting energies/forces on
  datasets like **Transition1x** (reaction paths), **SPICE**, and **ANI-1ccx**.
  Getting reactive intermediates into the training set is an **active-learning**
  loop (run MD → find high-uncertainty configs → label with QM → retrain) — the
  thing our per-member perturbation gestures at.
- **Autograd forces.** Real codes get forces by automatic differentiation
  (PyTorch); we wrote the gradient analytically to keep it transparent.
- **Electrostatic / polarizable embedding + long-range PME.** Mechanical
  embedding (ours) ignores how the MM charges polarize the ML region. Real hybrids
  use electrostatic embedding and Particle-Mesh-Ewald for long-range Coulomb —
  the catalog's noted hard problem at the boundary.
- **δ-ML / Δ-learning.** Often the NNP learns only the *correction* to a cheap
  baseline (semiempirical or a small basis DFT), which is easier to fit
  (Exercise 5).
- **Async CUDA streams.** Production overlaps the NNP forward pass and the MM
  evaluation in separate streams; here the *ensemble* is the parallel axis and
  both potentials live in one kernel (Exercise 4 adds the streams).
- **Scale.** Thousands of atoms in 3-D with neighbor lists and periodic boundary
  conditions, microsecond trajectories — vs. our 8 atoms, 1-D, hundreds of steps.

---

## References

- **Behler & Parrinello (2007)**, *Generalized Neural-Network Representation of
  High-Dimensional Potential-Energy Surfaces* — the per-atom-energy + symmetry-
  function idea this project implements in miniature.
- **MACE** (Batatia et al.) — <https://github.com/ACEsuit/mace> — modern
  higher-order equivariant NNP; study its message construction.
- **NequIP** (Batzner et al.) — the E(3)-equivariant NNP our scalar descriptor
  abstracts away.
- **OpenMM-ML** — <https://github.com/openmm/openmm-ml> — how an NNP is actually
  coupled to an MM environment (mechanical vs electrostatic embedding).
- **NNPOps** — <https://github.com/openmm/NNPOps> — CUDA-optimized neighbor lists
  and symmetry functions; the production version of our descriptor loop.
- **Transition1x** (Zenodo 5781475), **SPICE**, **ANI-1ccx** — the QM/DFT
  reference datasets you would train a real NNP on (see `data/README.md`).
- **Warshel & Levitt (1976)** — the original QM/MM partitioning idea (Nobel 2013).
