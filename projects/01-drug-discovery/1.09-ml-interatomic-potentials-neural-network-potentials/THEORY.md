# THEORY — 1.9 ML Interatomic Potentials (Neural Network Potentials)

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

To simulate how a drug binds its target, or how a protein folds, you need the
**potential energy** of the atoms as a function of their positions, `E(R)`. Take
the gradient and you get the **forces** `F = −∇E`; integrate the forces and you
get **molecular dynamics** (MD) — a movie of the molecule moving.

There are two classical ways to get `E(R)`, and they sit at opposite ends of an
accuracy/speed trade-off:

- **Quantum chemistry (DFT, CCSD(T)):** solve (an approximation of) the electronic
  Schrödinger equation. Very accurate, but `O(N³)`–`O(N⁷)` in the number of
  electrons — far too slow for long MD of large systems.
- **Classical force fields (AMBER, CHARMM):** hand-built spring-and-charge formulas.
  Cheap and fast, but fixed functional forms that cannot describe bond breaking
  (reactivity) and miss subtle quantum effects.

**Neural network potentials (NNPs)** are the modern middle ground: a neural network
is **trained** on quantum-chemistry reference energies/forces to *learn* `E(R)`. At
run time it is nearly as cheap as a force field but nearly as accurate as DFT — on
an A100, ~10 ns/day for a 500-atom system, ~100× faster than DFT. That speed is
what makes reactive drug-target simulations feasible.

The foundational idea (Behler & Parrinello, 2007) is the **additive atomic-energy
ansatz**: the total energy is a sum of per-atom contributions,

```
E(R) = Σ_i E_i ,
```

where each `E_i` is produced by a small neural network that sees only atom `i`'s
**local environment** (its neighbors within a cutoff). This single assumption gives
NNPs three superpowers: they are *size-extensive* (energy scales with system size),
*transferable* (a network trained on small molecules generalizes to bigger ones),
and — crucially for us — **embarrassingly parallel** (every `E_i` is independent).

This project implements that core pipeline in a deliberately **reduced** form so the
math stays visible: one atom type, **radial** symmetry functions only, and a small
fixed network. The structure is exactly ANI's; §7 lists what production NNPs add.

## 2. The math

**Inputs:** `n` atom positions `r_i ∈ ℝ³` (Angstrom). **Output:** the total energy
`E` (and, in a full NNP, forces `F_i = −∂E/∂r_i`).

### Step 1 — Atom-centered symmetry functions (the descriptor)

A neural network needs a fixed-length input that is **invariant** to translating
and rotating the whole molecule and to **permuting identical neighbors** (swapping
two equivalent neighbors must not change the energy). Raw coordinates are none of
these. Behler's *radial symmetry function* `G2` solves it. For atom `i` and shell
center `Rs_s`:

```
G2_s(i) = Σ_{j ≠ i,  r_ij < Rc}  exp( −η (r_ij − Rs_s)² ) · fc(r_ij)
```

- `r_ij = |r_i − r_j|` — interatomic distance (Angstrom). Using only distances makes
  the descriptor automatically translation- and rotation-invariant; summing over
  `j` makes it permutation-invariant.
- `Rs_s` — the center of shell `s` (we use 8 shells from 0.8 to 4.0 Å). Each shell
  asks "how much neighbor density sits near distance `Rs_s`?"
- `η` — Gaussian width (1/Å²); larger `η` = sharper shells. We use `η = 1.5`.
- `Rc` — cutoff radius (we use 5.0 Å). Neighbors beyond `Rc` are ignored → the model
  is short-ranged, which is what makes it parallel.
- `fc(r)` — the **cosine cutoff function**, the detail that makes everything smooth:

```
fc(r) = 0.5 · ( cos(π r / Rc) + 1 )   for r ≤ Rc ,    0 otherwise.
```

`fc` is `1` at `r=0` and tapers smoothly to `0` (with zero slope) at `r=Rc`. Without
it, an atom crossing the cutoff would make `E` jump discontinuously and forces blow
up. The descriptor of atom `i` is the vector `desc(i) = [G2_0(i), …, G2_{7}(i)]`.

### Step 2 — The per-atom neural network

A small multilayer perceptron (MLP) maps the descriptor to the atomic energy. With
two hidden layers of width `H` and `tanh` activations:

```
h1 = tanh( W1 · desc + b1 )      (H×D matrix W1, D = N_DESC = 8,  H = N_HID = 16)
h2 = tanh( W2 · h1   + b2 )      (H×H matrix W2)
E_i = w3 · h2 + b3               (1×H vector w3, scalar b3)
```

`tanh` is smooth and bounded — important because forces are derivatives of `E`, so
the activation must be differentiable. In a real NNP `W1,b1,…` are **learned** by
minimizing the error against DFT energies/forces; here they are **fixed**
(manufactured by a seeded PRNG, see §6) so the demo is reproducible and offline.

### Step 3 — Sum

```
E = Σ_{i=0}^{n−1} E_i .
```

## 3. The algorithm

```
build model: ACSF hyperparameters (Rc, η, Rs[]) + MLP weights        (once)
for each atom i in 0..n-1:                                            (PARALLEL)
    desc[0..7] = 0
    for each atom j != i:                                            (neighbor scan)
        r = |r_i - r_j|
        if r < Rc:
            f = fc(r)
            for each shell s:  desc[s] += exp(-η (r-Rs[s])^2) * f
    h1 = tanh(W1·desc + b1);  h2 = tanh(W2·h1 + b2);  E_i = w3·h2 + b3
E = sum_i E_i                                                        (reduction)
```

**Complexity.** The neighbor scan is the cost driver:

- Descriptor: `O(n)` per atom (brute-force scan of all others) × `N_DESC` shells →
  `O(n · N_DESC)` per atom, `O(n² · N_DESC)` total. A **cell list** (bucketing atoms
  into `Rc`-sized cells, scanning only the 27 neighboring cells) reduces this to
  `O(n · n̄)` where `n̄` is the average neighbor count — i.e. `O(n)`. We keep the
  brute-force scan because it is trivially correct and the demo is tiny (§7).
- MLP: a fixed `O(N_DESC·N_HID + N_HID²)` per atom — a few hundred FMAs, constant
  in `n`. Negligible next to the neighbor scan at scale.

**Arithmetic intensity / access pattern.** Each atom re-reads all `3n` coordinates
(the inner scan) but writes a single `E_i`. The work per atom is independent of
every other atom — there are *no* data dependencies between atoms — which is exactly
why this is a textbook data-parallel problem.

## 4. The GPU mapping

**Thread-to-data mapping.** *One thread owns one atom.* Thread
`i = blockIdx.x·blockDim.x + threadIdx.x` computes `E_i` end to end (its descriptor
+ its MLP forward), then a grid-stride loop lets a fixed grid cover any `n`. This is
the "independent jobs" pattern (PATTERNS.md row 1), the same shape as the 1.12
Tanimoto flagship — there, one thread scored one library molecule against the query;
here, one thread scores one atom against its environment.

**Launch configuration.** `block = 128` threads (a multiple of the 32-lane warp).
The per-atom MLP uses local arrays `desc[8]`, `h1[16]`, `h2[16]` of doubles, so each
thread is register/local-memory heavy; a moderate block size keeps register pressure
from crushing occupancy. `grid = ceil(n/128)`, capped at 1024 blocks with a
grid-stride loop covering the rest.

**Memory hierarchy — and why.**

- **Constant memory** holds the read-only model (`AcsfParams c_params`,
  `AtomicNet c_net`). Every thread reads the *same* weights and never writes them →
  the constant cache broadcasts one address to a whole warp in a single transaction,
  far cheaper than each thread issuing global loads. The model is small (~3.5 KB)
  and fixed-size at compile time, so it fits the 64 KB constant bank. This mirrors
  the constant-memory query in 1.12.
- **Global memory** holds the `3n` coordinates (read by the neighbor scan) and the
  `n` output energies (one write per thread). No `__restrict__` aliasing hazards;
  no shared memory needed (atoms are independent, nothing to cooperate on).
- **Registers / local memory** hold the per-thread `desc`, `h1`, `h2` and the MLP
  accumulators.

**No atomics.** The per-atom energies are written to distinct slots — no contention.
The final sum is done **on the host** in fixed atom order (see §5), not with a
parallel/atomic GPU reduction, so the total is deterministic and matches the CPU.

```
            grid of blocks (128 threads each), grid-stride over atoms
   ┌──────────────────────────────────────────────────────────────────┐
   │ thread i ─► atom i                                                 │
   │    read r_i, scan all r_j (global) ─► desc[8]                      │
   │    desc ─► [W1,b1]→tanh ─► [W2,b2]→tanh ─► [w3,b3] ─► E_i          │
   │                         ▲ weights broadcast from CONSTANT memory   │
   │    write E_i (global)                                              │
   └──────────────────────────────────────────────────────────────────┘
        d_eatom[0..n-1]  ──D2H──►  host sums in order  ──►  E = Σ E_i
```

**No CUDA library is used here on purpose** — the descriptor and MLP are
hand-written so nothing is a black box. A production NNP *would* lean on libraries:
the MLP is a batched GEMM (cuBLAS), and forces come from PyTorch's CUDA **autograd**
(reverse-mode automatic differentiation of `E` w.r.t. positions). Writing autograd
by hand means storing the forward activations and back-propagating
`∂E/∂h2 → ∂E/∂h1 → ∂E/∂desc → ∂E/∂r_j` through `tanh'` and the Gaussian — doable but
exactly the bookkeeping autograd automates (§7).

## 5. Numerical considerations

- **Precision: FP64 (double).** We use double throughout. The point of this project
  is an *exact* CPU-vs-GPU comparison, and double makes round-off negligible so the
  agreement is convincing. (Production inference often uses FP32/mixed precision for
  speed; that is an exercise.)
- **The one real source of CPU/GPU divergence: FMA.** The GPU contracts `a*b + c`
  into a single fused multiply-add (one rounding) where the host compiler may do two
  rounded operations. Over a short descriptor sum + a 2-hidden-layer MLP this differs
  by only ~`1e-15`. We *measure* it: the demo's stderr prints `max per-atom err`
  (~`1e-15`) and `total err` (~`1e-15`) — both far under the `1e-9` tolerance.
- **Determinism (PATTERNS.md §3).** Two deliberate choices keep stdout reproducible:
  1. Each atom's descriptor sums its neighbors in **ascending index order** on *both*
     host and device (the GPU thread runs the exact same loop), so per-atom values
     are identical, not merely close.
  2. The total is reduced **on the host in atom order**, never with float
     `atomicAdd` (which reorders and is non-associative in floating point). A
     fixed-order double sum is deterministic *and* matches the CPU's loop exactly.
- **No race conditions:** distinct output slots, no shared mutable state.

## 6. How we verify correctness

The oracle is `src/reference_cpu.cpp :: nnp_energy_cpu`: a plain serial loop over
atoms. The key design move (PATTERNS.md §2) is that **the per-atom physics lives in
one `__host__ __device__` header**, `src/nnp.h`. The CPU reference and the GPU kernel
*both* call the same `atomic_energy()` — same descriptor code, same MLP code — so
they do byte-for-byte identical arithmetic, and any difference is pure FP round-off.

- **Tolerance: `1e-9` (absolute, on energy).** Comfortably above the observed
  ~`1e-15` FMA divergence and far below any meaningful energy scale. The demo checks
  both the worst per-atom error and the total error against this bound.
- **Why this is convincing.** The CPU path is an *independent, obviously-correct*
  serial implementation. If a parallel bug existed (a missed neighbor, a wrong
  index, a race), the GPU total would drift from the serial total by far more than
  `1e-9`. Agreement across two implementations is strong evidence of correctness.
- **Edge cases.** The loader rejects empty/truncated files; the kernel's grid-stride
  loop + `if (i<n)` guard handle the ragged last block; `fc` returns 0 exactly at
  and beyond `Rc` (no neighbor double-counting at the boundary).
- **A second sanity anchor.** Because `make_synthetic.py` and `reference_cpu.cpp`
  share the *same* splitmix64 PRNG (documented in both), the synthetic inputs and the
  weight-generation rule are reproducible and inspectable in either language.

## 7. Where this sits in the real world

This is a **reduced-scope teaching version** (CLAUDE.md §13). Production NNPs add:

- **Angular descriptors (G4/G5).** Real ACSF includes three-body terms depending on
  the angle `θ_jik` — essential to distinguish, say, linear vs. bent geometries that
  radial terms alone cannot. ANI uses both radial and angular symmetry functions.
- **Multiple elements.** Each element gets its own network and its own descriptor
  channels (H sees different chemistry than O). We use a single untyped atom.
- **Learned weights.** Real weights are fit to **ANI-1ccx / SPICE / rMD17** energies
  *and forces* by gradient descent (TorchANI, NequIP, MACE). Ours are manufactured.
- **Analytic forces via autograd.** MD needs `F = −∇E`; frameworks get it for free
  by reverse-mode autodiff of the energy graph. We compute energy only.
- **Equivariant message passing.** State-of-the-art models (**NequIP**, **MACE**)
  replace hand-designed symmetry functions with **E(3)-equivariant** features that
  transform correctly under rotation, learning richer, more data-efficient
  descriptors via message passing on the neighbor graph.
- **Neighbor lists + PBC.** Real codes use cell lists / Verlet lists (O(n) not O(n²))
  and periodic boundary conditions for condensed phases. We brute-force the scan.
- **Performance engineering.** `torch.compile`/TorchScript for inference, multi-GPU
  training via DDP, custom fused CUDA kernels (MACE) for the equivariant tensor
  products.

What *does* transfer directly from this project: the additive-atomic-energy ansatz,
the symmetry-function idea (invariance by construction), the per-atom-network
structure, and — most importantly for this repo — the **one-thread-per-atom GPU
mapping** with a constant-memory model and a deterministic host-side reduction.

---

## References

- **Behler & Parrinello**, *Generalized Neural-Network Representation of
  High-Dimensional Potential-Energy Surfaces*, PRL 98, 146401 (2007) — the
  additive atomic-energy ansatz this project implements.
- **Behler**, *Atom-centered symmetry functions…*, J. Chem. Phys. 134, 074106 (2011)
  — the `G2`/`G4` symmetry functions and the cosine cutoff `fc`.
- **Smith, Isayev, Roitberg**, *ANI-1*, Chem. Sci. 8, 3192 (2017) — the ANI model;
  see **TorchANI** (<https://github.com/aiqm/torchani>) for the production code.
- **Batzner et al.**, *NequIP* (Nat. Commun. 2022) and **Batatia et al.**, *MACE*
  (NeurIPS 2022) — E(3)-equivariant NNPs (<https://github.com/mir-group/nequip>,
  <https://github.com/ACEsuit/mace>); study what equivariance buys over symmetry
  functions.
- **TorchMD-Net** (<https://github.com/torchmd/torchmd-net>) — equivariant NNPs with
  a GPU-optimized neighbor list; study the neighbor-list construction we omit.
