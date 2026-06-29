# THEORY — 1.23 QM/MM Molecular Dynamics

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

This project is a **reduced-scope teaching version** (CLAUDE.md §13) of QM/MM
molecular dynamics. It keeps the conceptual skeleton of QM/MM — a quantum force on
a reactive coordinate, a classical environment, electrostatic embedding, and a
Verlet integrator — but replaces the expensive electronic-structure solver with an
**exactly solvable 2×2 model** so the entire loop fits in one readable header.

---

## 1. The science

**The problem.** Many of the most important events in biochemistry are
*chemical reactions*: an enzyme cleaves a bond, a covalent drug attacks a
catalytic cysteine, a proton hops along a hydrogen-bonded wire. Classical
molecular dynamics (springs-and-charges force fields) **cannot** describe these,
because a fixed bonded topology has no way to break or make bonds. You need
**quantum mechanics** — the electrons — to describe bond rearrangement.

But quantum mechanics is expensive: solving the electronic structure of a whole
solvated protein is hopeless. The **QM/MM** insight (Warshel & Levitt, 1976;
Nobel Prize 2013) is that the *chemistry* happens in a small **reactive region**
(the substrate plus a few catalytic residues, ~50–200 atoms). Treat **only that
region quantum-mechanically** (QM), and treat the thousands of surrounding atoms
with a cheap **classical force field** (MM). The two regions are coupled — most
importantly, the MM environment's charges create an electrostatic field that
**polarizes** the QM region. That coupling is called **electrostatic embedding**,
and it is exactly how a charged residue or an ordered water tilts a reaction's
energy landscape to catalyze it.

**The reaction we model.** The textbook QM/MM use case is **proton transfer**: a
proton (H⁺) moves from a donor atom to an acceptor atom. We model the proton's
position with a single reaction coordinate `x`. The electronic structure has two
relevant **diabatic** (character-preserving) states:

- |L⟩ — the proton bonded to the **donor** (it sits in a well at `x = X_L < 0`),
- |R⟩ — the proton bonded to the **acceptor** (a well at `x = X_R > 0`).

The MM environment's field decides which side is energetically favored. With no
field the two wells are symmetric (no net reaction); a field tilts the surface so
the proton transfers. That is the whole story this project tells.

---

## 2. The math

**Diabatic states.** Each diabatic state is a harmonic well in the proton
coordinate. Including the electrostatic-embedding shift from a uniform MM field
`F` (units energy/length) acting on the proton's charge `q`:

```
e_L(x) = ε₀ + ½ k (x − X_L)²  +  q·F·x
e_R(x) = ε₀ + ½ k (x − X_R)²  +  q·F·x
```

with well curvature `k = KWELL`, minima `X_L, X_R`, charge `q = QPROTON`. The term
`q·F·x` is the embedding: a linear potential that lowers one side relative to the
other (charge × field = energy/length, times position = energy).

**The 2×2 Hamiltonian.** The two diabatic states are coupled by an **electronic
coupling** `β = COUPLING` (the tunneling/resonance matrix element). In the
{|L⟩, |R⟩} basis the electronic Hamiltonian is the real-symmetric matrix

```
        ⎡ e_L(x)   β     ⎤
H(x) =  ⎢                ⎥
        ⎣ β        e_R(x)⎦
```

**Adiabatic surfaces.** The Born–Oppenheimer surfaces the nuclei move on are the
**eigenvalues** of `H`. For a symmetric 2×2 matrix these are available in closed
form:

```
E±(x) = ½(e_L + e_R)  ±  √( [½(e_L − e_R)]² + β² )
```

The **ground-state** surface is the lower root `E₋(x)`. The quantity under the
root, doubled, is the **adiabatic gap** `Δ(x) = 2√([½(e_L−e_R)]² + β²) ≥ 2β`: the
energy separation between ground and excited surfaces. It is smallest (`= 2β`)
where the diabatic states are degenerate (`e_L = e_R`, i.e. over the barrier).

**The barrier.** At `F = 0` the diabatic curves cross at `x = 0` with energy
`½ k X_R²`. Coupling lowers the adiabatic ground state there by `β`, so the
barrier height is roughly

```
ΔE‡ ≈ ½ k X_R² − β
```

With the project's constants (`k = 30, X_R = 0.6, β = 2`): `ΔE‡ ≈ 3.4` — a genuine
double well. Choosing `β` *small* compared to `½ k X_R²` is what keeps a real
barrier; a large `β` collapses the two wells into one (nothing to transfer).

**Nuclear dynamics.** The proton (mass `m = PROTON_MASS`) moves under the total
QM/MM force:

```
F_tot(x) = −dE₋/dx  −  k_MM · x        (QM force + classical MM tether)
```

where the analytic QM gradient (differentiating `E₋`) is

```
dE₋/dx = ½(e_L' + e_R')  −  [½(e_L−e_R) · ½(e_L'−e_R')] / √([½(e_L−e_R)]² + β²)
e_L' = k(x − X_L) + qF,   e_R' = k(x − X_R) + qF
```

and `−k_MM·x` is a weak harmonic tether standing in for the protein/solvent cage
(the classical "MM force"). Newton's equation `m ẍ = F_tot(x)` is integrated in
time.

**Symbols.** `x` proton coordinate (length); `F` MM field (energy/length); `q`
proton charge; `k` diabatic curvature (energy/length²); `X_L,X_R` well minima;
`β` electronic coupling (energy); `m` proton mass; `k_MM` MM stiffness; `dt`
timestep; everything in a self-consistent **model** unit system (not real atomic
units — see §5).

---

## 3. The algorithm

For one trajectory with initial `(x₀, v₀)` and field `F`:

```
a ← F_tot(x₀)/m                         # seed acceleration (one QM solve)
for s in 1..steps:                      # velocity Verlet
    x ← x + v·dt + ½ a·dt²              # drift with current acceleration
    a_new ← F_tot(x)/m                  # NEW QM force  ← the per-step quantum solve
    v ← v + ½ (a + a_new)·dt            # kick with averaged acceleration
    a ← a_new
    track min adiabatic gap, time on product side (x>0)
report final x, final energy, min gap, % product, transferred?
```

Each `F_tot` evaluation builds the 2×2 Hamiltonian and its analytic gradient —
that is the "QM Hamiltonian evaluation at every MD step" the catalog names as the
bottleneck, here made `O(1)`.

**Complexity.** One trajectory is `O(steps)` (a constant amount of work per step).
The **ensemble** of `M = nf·nx` trajectories is `O(M · steps)` total. Serially the
CPU walks all `M` trajectories one after another; in parallel the **work** is the
same `O(M·steps)` but the **depth** (critical path) is just `O(steps)` — a single
trajectory's length — because all `M` run concurrently. That gap between work and
depth is precisely what the GPU exploits.

**Arithmetic intensity.** Very high: each step is a handful of multiplies, adds,
and one `sqrt`, all in registers, with essentially **zero global-memory traffic**
during the loop. This kernel is **compute/latency-bound**, not bandwidth-bound —
the opposite of a stencil or a reduction.

---

## 4. The GPU mapping

**Thread-to-data mapping.** One **thread = one trajectory**. Thread
`idx = blockIdx.x·blockDim.x + threadIdx.x` owns ensemble member `idx`; it reads
its `(field, x0)` from the sweep grid via `member_params()` and runs the entire
Verlet loop in registers, then writes one `TrajResult` to `out[idx]`.

```
ensemble of M = nf*nx trajectories  ──►  M GPU threads

 grid:   ceil(M / 128) blocks
 block:  128 threads
         ┌──────────────── block b ────────────────┐
         │ t0   t1   t2   ...               t127    │   each thread:
         │ │    │    │                        │     │     x,v,a in registers
         │ run  run  run                      run   │     5000-step Verlet loop
         │ Verlet loops independently (no comms)    │     1 write to out[idx]
         └──────────────────────────────────────────┘
```

**Launch configuration.** `THREADS_PER_BLOCK = 128`. This kernel is
**register-heavy**: each thread holds the integrator state plus the Verlet
temporaries, so register pressure — not warp count — limits occupancy. 128 keeps
four warps resident per block while leaving registers for the time loop. (256 also
works; profile per GPU.) Grid size is the usual ceiling division so the ragged
last block is covered, with an `if (idx >= M) return;` guard.

**Memory hierarchy.**
- **Registers** — the entire hot loop lives here (`x, v, accel`, the Verlet
  temporaries, the running accumulators). This is why the kernel is fast and why
  register count is the occupancy knob.
- **Global memory** — touched only twice per thread: read the small
  `EnsembleConfig` (passed by value, so it lands in constant/parameter space) and
  write one `TrajResult` at the end. No coalescing concerns inside the loop.
- **Shared memory / atomics** — **none.** Trajectories are independent, so there
  is nothing to share or to reduce across threads.

**No CUDA library — and why.** The quantum step is a **symmetric 2×2
diagonalization**, which has an exact closed-form solution (the `±√…` formula in
§2). So we hand-roll it in a few FLOPs and link only `cudart_static`. **When would
you reach for a library?** The moment the QM region is more than two states. A real
QM/MM step diagonalizes (or SCF-iterates) a Hamiltonian/Fock matrix of size
`N_basis × N_basis` with `N_basis` in the hundreds-to-thousands — that is a dense
**eigenproblem** (`cuSOLVER` `Dsyevd`, exactly the tool flagship 2.06 uses) or a
sequence of **GEMMs** (`cuBLAS`) to build and contract the density matrix. Writing
a general symmetric eigensolver by hand (Householder tridiagonalization + QR/divide-
and-conquer) is a substantial undertaking; for `N = 2` it is one square root, which
is why we can stay library-free here and keep CPU/GPU bit-comparable.

---

## 5. Numerical considerations

**Precision: FP64 throughout.** Velocity Verlet accumulates `x` and `v` over
thousands of steps; in FP32 the rounding would visibly drift the energy and (worse)
make the CPU and GPU disagree. We use `double` everywhere in `qmmm.h`, which keeps
per-step rounding near `1e-16` and the GPU-vs-CPU difference at the level reported
below.

**Why velocity Verlet.** It is **symplectic** and **time-reversible**, so it
conserves a "shadow" energy and does not secularly drift the total energy the way
naive Euler does. On the committed sample, the no-field, bottom-of-well trajectory
conserves total energy (kinetic + QM/MM potential) to a spread of ~`9e-7` over 5000
steps — a clean demonstration of the integrator's quality. It is **2nd-order**:
halving `dt` cuts the energy drift ~4× (Exercise 2). At very strong fields the
proton makes large, fast excursions and the fixed `dt` resolves them less well, so
the conserved-energy spread grows to ~`1e-3` — still physically negligible, and an
honest illustration of why production code uses a smaller `dt` for energetic motion.

**Determinism.** There are **no atomics and no parallel reductions** in this
kernel — each thread's trajectory is fully independent and writes its own output
slot. So there is no floating-point-reordering nondeterminism (contrast the
Monte-Carlo tally in flagship 5.01, which must accumulate in integers to stay
deterministic). Every thread executes the identical straight-line Verlet loop;
divergence is limited to cheap predicated updates (the min-gap and product-side
branches). The program prints results at **fixed precision** to stdout, so the
demo's diff is byte-stable run to run.

**Race conditions.** None possible: distinct threads write distinct `out[idx]`
elements; there is no aliasing (`__restrict__` documents this) and no shared state.

---

## 6. How we verify correctness

**Two independent implementations of the same math.** The CPU reference
(`reference_cpu.cpp`) loops `qmmm::integrate_trajectory` over all members; the GPU
kernel calls the *same* `qmmm::integrate_trajectory` from one thread each. Because
the per-step physics and the Verlet integrator are `__host__ __device__` inline
functions in **one shared header** (`qmmm.h`, PATTERNS.md §2), the CPU and GPU
execute the identical sequence of double-precision operations.

**Tolerance.** We compare every continuous per-member output (`final_x`,
`final_energy`, `min_gap`, `frac_product`) and take the worst absolute difference.
The tolerance is **`1e-9`**; the measured worst diff is ~`1e-12`. The residual is
not a bug — over thousands of steps the GPU's **fused multiply-add** (FMA) contracts
`a*b + c` into one rounding, while the host compiler may emit a separate multiply
and add, so the two diverge by a few ULPs per step that accumulate to ~`1e-12`
(PATTERNS.md §4, "machine-precision band"). We verify to a tight but non-zero
tolerance rather than demanding bit-identity, and we print the actual diff to
stderr so the claim is auditable.

**A second, physical check.** Beyond CPU==GPU agreement, the *result* is
independently sensible: at `field = 0` every trajectory stays trapped
(`transferred = 0`), and `min_gap` bottoms out at exactly `2β = 4.0` whenever the
proton reaches the barrier — both predictions of §2, recovered by the simulation.

**Edge cases.** `√(diff² + β²) ≥ β > 0` always, so the gap never vanishes and the
force has no singularity; the loader rejects non-positive `dt`/`steps`/`nf`/`nx`;
the kernel guards the ragged last block.

---

## 7. Where this sits in the real world

Production GPU QM/MM (the catalog's "Prior art") differs in **what happens inside
one step**, not in the outer loop:

- **The QM solve.** Instead of a 2×2 model, real codes solve the electronic
  structure of the whole QM region with **DFT** (B3LYP/PBE) or a semi-empirical
  method (**GFN2-xTB**). The cost is dominated by **electron-repulsion integrals
  (ERIs)** and the self-consistent-field (SCF) iteration — this is where
  **TeraChem** and **AMBER+QUICK** put their CUDA kernels (ERI evaluation, Fock
  builds, density-matrix GEMMs). The "diagonalize the Hamiltonian" step becomes a
  dense **`cuSOLVER` eigensolve** of an `N_basis × N_basis` matrix.
- **The coupling.** Real electrostatic embedding sums Coulomb interactions between
  every QM electron/nucleus and **thousands of MM point charges** (often with
  Particle-Mesh Ewald), not a single uniform field. Covalent bonds cut by the
  QM/MM boundary are healed with **link atoms** (the ONIOM scheme).
- **The MM side.** The classical region is a full biomolecular force field run on
  the GPU (**`pmemd.cuda`**, OpenMM), with its own bonded + Lennard-Jones + PME
  electrostatics over the whole system.
- **The orchestration.** Because the QM and MM engines are often separate programs,
  production setups overlap them with **CUDA streams** and asynchronous GPU↔CPU
  transfer (e.g. TeraChem's TCPB protocol), hiding the QM solve behind MM work.
- **Sampling.** To get a *rate* or a free-energy barrier you run **thousands of
  trajectories** with a thermostat and enhanced sampling (umbrella sampling,
  metadynamics) — which is exactly the ensemble dimension this project parallelizes,
  just with a real force at each step.

What we keep, faithfully, is the **shape**: build a quantum Hamiltonian that the MM
environment polarizes, take its ground-state energy and force, add the classical
force, advance with Verlet, and sample an ensemble on the GPU. Swap the 2×2 solve
for a DFT step and the toy becomes the tool.

---

## References

- A. Warshel, M. Levitt, *Theoretical studies of enzymic reactions*, J. Mol. Biol.
  103 (1976) 227 — the original QM/MM idea (2013 Nobel Prize in Chemistry).
- H. M. Senn, W. Thiel, *QM/MM Methods for Biomolecular Systems*, Angew. Chem. Int.
  Ed. 48 (2009) 1198 — the standard QM/MM review (embedding, link atoms, ONIOM).
- **AMBER + QUICK** — <https://github.com/merzlab/QUICK>: how a GPU DFT engine
  supplies the QM force to MD each step.
- **TeraChem** — <https://www.petachem.com>: GPU DFT and the TCPB client/server
  protocol for QM/MM.
- **OpenMM + PySCF** — <https://github.com/openmm/openmm>: a readable reference for
  the embedding/bookkeeping side of QM/MM.
- **CP2K** — <https://github.com/cp2k/cp2k>: large-scale, periodic GPU QM/MM.
- D. Frenkel, B. Smit, *Understanding Molecular Simulation* — velocity Verlet,
  symplectic integrators, and why energy is conserved (Ch. 4).
