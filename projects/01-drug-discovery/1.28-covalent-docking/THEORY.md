# THEORY — 1.28 Covalent Docking

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use. The geometry, force field, and pocket
> here are **synthetic** and deliberately simplified._

---

## 1. The science

Most drugs bind their target **reversibly**: the molecule drifts into a pocket,
is held by weak forces (van der Waals contacts, hydrogen bonds, electrostatics),
and can drift back out. **Covalent inhibitors** are different — they form an
actual **chemical bond** to the protein, usually to the nucleophilic side chain
of a specific residue: most often a **cysteine** (its sulfur, "S-gamma"), but
sometimes serine, lysine, or tyrosine. That bond can be effectively permanent.

Why do this? A covalent bond gives durable, often very selective inhibition. The
clinical track record is real and current:

- **KRAS(G12C)** — sotorasib and adagrasib bond the mutant cysteine of a cancer
  target that was "undruggable" for decades.
- **BTK** — ibrutinib bonds a cysteine in Bruton's tyrosine kinase (leukemias).
- **EGFR** — afatinib/osimertinib bond a cysteine in the EGFR kinase (lung cancer).

The drug carries a **warhead**: a mildly reactive chemical group (acrylamide,
chloroacetamide, …) positioned to react with the target residue once the rest of
the molecule is correctly placed. **Docking** a covalent inhibitor — predicting
the bound 3-D pose computationally — is therefore a **two-stage** problem:

1. **Pre-reaction (non-covalent) placement.** Position the warhead so it can reach
   the reactive residue, exactly as in ordinary docking.
2. **Post-reaction scoring with the bond formed.** *Enforce* the covalent bond's
   ideal geometry (length, angle), then sample the remaining flexible parts of the
   ligand and score how well the whole molecule fits the pocket.

Stage 2 is where the GPU earns its keep. Once the warhead is anchored, the rest
of the ligand still has **rotatable bonds** (torsions). Every combination of
torsion angles is a candidate pose, and the number of combinations grows
**exponentially** with the number of rotatable bonds — the "curse of
dimensionality". Each candidate is scored independently, so the search is
embarrassingly parallel: **one GPU thread per conformation**.

This project implements a **reduced-scope, didactic** version of exactly that
stage-2 search. See §7 for what a production covalent docker adds.

## 2. The math

### Geometry: the flexible ligand as a torsion chain

The warhead atom (the **anchor**, position $\mathbf{a}$) is fixed: stage 1 chose
it, and it is covalently bonded to the cysteine sulfur $\mathbf{s}$. The ligand
is modeled as a short open chain of $N_{\text{atoms}}=4$ atoms past the anchor,
joined by rigid bonds of length $\ell$ at a fixed valence angle $\beta$. The
**degrees of freedom** are the $N_\tau = 3$ **torsion angles**
$\boldsymbol{\theta} = (\theta_0, \theta_1, \theta_2)$, one per rotatable bond.

A conformation's atom positions come from **forward kinematics**: starting at the
anchor with a reference bond direction $\hat{\mathbf{d}}_0$, place each next atom
by (a) bending the current bond direction by the valence angle and (b) twisting it
about the previous bond axis by that bond's torsion. The twist uses **Rodrigues'
rotation formula**, rotating a vector $\mathbf{v}$ by angle $\theta$ about a unit
axis $\hat{\mathbf{k}}$:

$$
\mathbf{v}_{\text{rot}} = \mathbf{v}\cos\theta + (\hat{\mathbf{k}}\times\mathbf{v})\sin\theta + \hat{\mathbf{k}}\,(\hat{\mathbf{k}}\cdot\mathbf{v})(1-\cos\theta).
$$

### Energy (the docking score)

We score a conformation with a small physics-style energy. Lower energy = better
pose. Two parts:

**(a) Covalent constraint** — a harmonic penalty that keeps the warhead at the
ideal bond geometry to the sulfur:

$$
E_{\text{cov}} = \tfrac12 k_b\,(|\mathbf{a}-\mathbf{s}| - \ell_0)^2 + \tfrac12 k_a\,(\phi - \phi_0)^2,
$$

where $\ell_0 \approx 1.81\,\text{Å}$ is the ideal C–S length, $\phi$ is the
Sγ–anchor–first-atom angle with ideal $\phi_0 \approx 109.47^\circ$ (sp³
tetrahedral), and $k_b, k_a$ are spring constants. **This term is what makes
covalent docking different from ordinary docking** — it is the "covalent bond
geometry constraint" of the catalog.

**(b) Nonbonded interaction** — how well the flexible ligand fits the rigid pocket.
For each ligand atom $i$ and pocket atom $j$ at distance $r_{ij}$:

$$
E_{\text{nb}} = \sum_{i,j} \Big[\, 4\varepsilon_{ij}\Big(\big(\tfrac{\sigma_{ij}}{r_{ij}}\big)^{12} - \big(\tfrac{\sigma_{ij}}{r_{ij}}\big)^{6}\Big) + k_C\,\frac{q_i q_j}{r_{ij}} \,\Big].
$$

The first bracket is the **Lennard-Jones 12-6** potential: the $r^{-12}$ term is a
steep repulsive wall (atoms cannot overlap), the $-r^{-6}$ term is the attractive
van der Waals well (shape complementarity), with minimum at $r = 2^{1/6}\sigma$.
The second is **Coulomb electrostatics** ($k_C = 332.06\ \text{kcal·Å·mol}^{-1}\text{e}^{-2}$).
Pair parameters use the **Lorentz–Berthelot** combining rules:
$\sigma_{ij} = \tfrac12(\sigma_i+\sigma_j)$, $\varepsilon_{ij} = \sqrt{\varepsilon_i\varepsilon_j}$.

### The objective

Search the **torsion grid** — each $\theta_j$ sampled at $G=36$ values over
$[0,2\pi)$ — for the conformation of least total energy:

$$
\boldsymbol{\theta}^\star = \arg\min_{\boldsymbol{\theta}}\; E_{\text{cov}}(\boldsymbol{\theta}) + E_{\text{nb}}(\boldsymbol{\theta}).
$$

The grid has $M = G^{N_\tau} = 36^3 = 46\,656$ conformations.

## 3. The algorithm

```
for each conformation id in [0, M):          # M = G^Nτ
    (θ0,θ1,θ2)  <- decode id  (mixed-radix base G)
    positions   <- forward_kinematics(anchor, θ)     # Rodrigues twists
    E[id]       <- E_cov(positions) + E_nb(positions, pocket)
best            <- argmin_id E[id]
```

- **Decode** uses the mixed-radix identity $id = \sum_j a_j G^j$, so thread/loop
  index `id` maps deterministically to a unique angle triple — and the **same**
  mapping is used by the CPU and GPU so "conformation `id`" means the same pose on
  both.
- **Complexity.** Each conformation costs $O(N_{\text{atoms}}\cdot N_{\text{pocket}})$ —
  here $4\times 6 = 24$ pair evaluations plus the kinematics. Total serial work is
  $O(M\cdot N_{\text{atoms}}\cdot N_{\text{pocket}})$. The crucial fact is $M$ is
  **exponential** in $N_\tau$: add one rotatable bond and the work multiplies by
  $G=36$. That exponential is why real flexible docking is expensive — and why a
  parallel machine helps.
- **Arithmetic intensity.** Each thread reads only the small shared `DockProblem`
  (constant for all threads) and writes one double. There is essentially no input
  bandwidth — the kernel is **compute-bound** on transcendentals (sin/cos/acos/
  sqrt) and the LJ powers. That is the ideal regime for a GPU.

## 4. The GPU mapping

**Pattern:** *score N independent candidates* — the same shape as project 1.12
(Tanimoto). Each conformation is independent, so:

- **Thread-to-data map.** Global thread index `id = blockIdx.x*blockDim.x +
  threadIdx.x` owns conformation `id`. A **grid-stride loop** (`id += blockDim.x*
  gridDim.x`) lets a fixed, modest grid cover all $M$ conformations, even if $M$
  grows.
- **Launch config.** `block = 256` threads (a multiple of the 32-lane warp, 8 warps
  to hide latency); `grid = min(ceil(M/256), 1024)` blocks. Because the kernel is
  compute-bound, the exact block size barely matters; 256 keeps register pressure
  low so occupancy stays high.
- **Memory hierarchy.** The `DockProblem` is passed **by value** as a kernel
  argument — it lands in each thread's constant/parameter space, read-only and
  identical for all threads, so there is no global-memory traffic for inputs. Each
  thread keeps its 4 atom positions in **registers/local memory** (small,
  fixed-size arrays) and writes one double to global `out[id]`. **No shared
  memory, no atomics** — outputs are fully independent.
- **No library call.** This kernel is hand-written; there is no FFT/GEMM/sort step
  to delegate. (A larger docker would use cuRAND for stochastic search and
  Thrust/CUB to reduce the energy array on-device — see §7.)

```
   conformation grid (M = 46656)
   id:  0    1    2   ...                      M-1
        │    │    │                             │
   ┌────▼────▼────▼─────────────── ... ─────────▼────┐   one thread per id
   │ t0   t1   t2                              t...   │   (grid-stride loop)
   └──┬───────────────────────────────────────────┬──┘
      │ each thread:                               │
      │   decode id -> (θ0,θ1,θ2)                  │
      │   forward kinematics (Rodrigues) -> atoms  │
      │   E_cov + E_nb -> out[id]                  │
      ▼                                            ▼
   out[0..M-1] (device) ──cudaMemcpy──> host ──argmin──> best pose
```

The final **argmin is done on the host** after copying the energy array back. We
*could* use a device `atomicMin`, but a floating-point atomic reduction is
order-dependent in its tie-breaking and therefore **nondeterministic**; reducing
on the host keeps stdout byte-identical run to run (docs/PATTERNS.md §3).

## 5. Numerical considerations

- **Precision: FP64 throughout.** Energies span small differences near a smooth
  minimum, and the trig/LJ math is sensitive; double precision keeps the landscape
  faithful and — crucially — lets the CPU and GPU agree to round-off.
- **CPU/GPU parity by construction.** The per-conformation physics lives in **one
  `__host__ __device__` header** (`src/docking.h`). The CPU reference loops it; the
  kernel calls it from one thread. Identical operations in identical order ⇒
  bit-for-bit (modulo one FMA rounding) identical energies. The measured worst-case
  difference on the sample is `~2e-15 kcal/mol` — that is **machine precision**.
- **Why the FMA caveat is tiny here, and why it would not always be.** The GPU may
  contract `a*b+c` into a single fused-multiply-add (one rounding) where the host
  does two. On a **smooth** landscape that perturbs each energy by a few ULP. We
  deliberately engineered the pocket so the ligand can **never clash** (every
  pocket atom sits at the LJ minimum distance, ≈3.82 Å, from the closest reachable
  ligand point), keeping all energies $O(1)$. If a conformation *did* drive
  $r\to 0$, the $r^{-12}$ term would reach $\sim 10^{18}$, and a few-ULP relative
  FMA difference there becomes an **absolute** difference of $\sim 10^{4}$ — a real
  trap we avoided by design, and a worthwhile lesson (see Exercises).
- **No atomics, no races.** Each thread writes a distinct `out[id]`; the reduction
  is a separate, deterministic host pass with ties broken by lowest `id`.
- **Determinism.** The torsion grid is a fixed uniform grid (not random), so the
  search and its result are fully reproducible; stdout is byte-identical every run.

## 6. How we verify correctness

`src/reference_cpu.cpp` is an independent, obviously-correct serial scan that fills
the **same** energy array the GPU produces. `main.cu` compares them
element-by-element with `max_abs_err` and asserts it is below the **tolerance
`1e-6 kcal/mol`** (chosen well above the ~`2e-15` FMA noise but far below any
physically meaningful energy difference — docs/PATTERNS.md §4). Both arrays then
go through the **same** `argmin_energy`, so the reported docked pose is identical.

Two stronger, science-level checks beyond "CPU == GPU":

1. **Covalent geometry sanity.** The program prints the warhead–Sγ bond length
   (1.810 Å, exactly the ideal) and the docked pose's torsions — you can confirm
   the constraint is honored.
2. **A real, deep minimum exists.** The best energy is **negative**
   (`−2.347 kcal/mol`) and ~25 000 of the 46 656 conformations are favorable
   (negative), so the landscape is a genuine well, not numerical noise.

Because two *independent* implementations (a plain CPU loop and a parallel CUDA
kernel) agree to machine precision **and** recover a sensible minimum, we believe
the result.

## 7. Where this sits in the real world

This is a **reduced-scope teaching model**, not a docking engine. Production
covalent docking differs in almost every component:

- **Sampling.** Real dockers do not enumerate a dense grid; they use **stochastic
  global search** (genetic algorithms / Lamarckian GA in AutoDock-GPU, Monte Carlo
  + gradient minimization in others) over **all** ligand torsions plus translation
  and rotation, with hundreds of rotatable-bond systems. The grid here is a
  didactic stand-in that makes the parallelism obvious.
- **Energy / scoring.** Real tools use full force fields and empirical scoring
  (AutoDock4/Vina terms, **MM-GBSA** rescoring including solvation), or learned
  CNN scoring (**GNINA**). We use a toy LJ + Coulomb + harmonic constraint.
- **Two-stage protocol.** Production covalent docking (Schrödinger **CovDock**,
  **AutoDock-GPU** covalent mode) explicitly models the warhead **reaction
  chemistry** via reaction SMARTS, samples the pre-reaction pose, forms the bond,
  then minimizes and rescores. We collapse stage 1 into a fixed anchor and focus
  on stage 2's flexible search.
- **Structures.** Real input is a prepared protein (protonation, the actual
  cysteine, the pocket) from the PDB, not 6 synthetic points.

**Starter tools to study** (catalog "Prior art"): **AutoDock-GPU** (the GA on the
GPU, and its covalent mode), **GNINA** (CNN scoring with covalent options),
**Uni-Dock** (a modern GPU docking engine), and the **CovDocker** 2025 benchmark
(a learned approach + dataset).

What carries over from this project to the real thing: the **independent-candidate
GPU pattern**, the **shared host/device scoring core** for exact verification, the
**covalent constraint penalty** as a distinct energy term, and the discipline of
keeping the energy landscape numerically well-behaved.

---

## References

- **AutoDock-GPU** — https://github.com/ccsb-scripps/AutoDock-GPU — the canonical
  GPU docking engine (Lamarckian GA on the GPU); read its covalent-docking mode and
  its per-individual parallelism.
- **GNINA** — https://github.com/gnina/gnina — CNN-scored docking with covalent
  options; study how a learned scoring function replaces hand-tuned terms.
- **Uni-Dock** — https://github.com/dptech-corp/Uni-Dock — a modern GPU docking
  engine; good for seeing batched, high-throughput GPU docking design.
- **CovDocker (2025)** — arXiv:2506.21085 — a deep-learning covalent-docking
  benchmark; verify the released code/data URL before relying on it.
- **Lennard-Jones & Lorentz–Berthelot rules** — any molecular-mechanics text
  (e.g. Leach, *Molecular Modelling: Principles and Applications*) for the energy
  terms and combining rules used in §2.
- **Rodrigues' rotation formula** — the standard axis–angle rotation used for the
  forward kinematics in §2 (`rotate_about_axis` in `src/docking.h`).
