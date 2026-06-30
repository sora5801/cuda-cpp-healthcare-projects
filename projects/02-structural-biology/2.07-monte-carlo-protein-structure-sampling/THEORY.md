# THEORY — 2.7 Monte Carlo Protein Structure Sampling

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See
> [README.md](README.md) for the quick tour and build steps.
>
> _Educational only — not for clinical use. This is a deliberately **reduced-scope
> teaching model** (the 2-D HP lattice protein); §7 explains how real engines
> differ._

---

## 1. The science

Proteins are linear chains of amino-acid residues that spontaneously **fold**
into a compact 3-D shape, and that shape determines what the protein does. A
central driving force of folding is the **hydrophobic effect**: water-fearing
("hydrophobic") residues cluster together in a buried core, away from the
surrounding water, while water-loving ("polar") residues stay on the surface.
Predicting the folded shape from the sequence is one of biology's grand problems.

One family of methods for exploring the space of possible shapes is **Monte Carlo
(MC) sampling**: repeatedly propose a small random change to the structure
(rotate a backbone angle, move a residue), accept or reject it by a physically
motivated rule, and repeat millions of times. Over many steps the simulation
spends most of its time in low-energy (likely) conformations, sampling the
*Boltzmann distribution* of structures.

This project models the science with the classic **HP lattice protein** of Lau &
Dill (1989) — the simplest model that still captures hydrophobic collapse:

- The chain lives on the 2-D integer grid `Z²`; each residue occupies one lattice
  cell, and consecutive residues are lattice neighbours (a self-avoiding walk).
- Each residue is just one of two types: **H** (hydrophobic) or **P** (polar).
- The energy rewards bringing H residues next to each other:
  `E = -ε · (number of non-bonded H–H contacts)`.

Minimizing `E` (= maximizing buried H–H contacts) is a stand-in for the
hydrophobic collapse that folds real proteins. Despite its simplicity the HP
model is **NP-hard** to optimize and is a standard testbed for folding/sampling
algorithms — perfect study material.

## 2. The math

**State.** A conformation is the set of lattice coordinates
`{(xᵢ, yᵢ)}` for residues `i = 0 … n-1`, subject to two constraints:
- *connectivity*: `|(xᵢ,yᵢ) − (xᵢ₊₁,yᵢ₊₁)| = 1` (chain bonds are unit lattice steps);
- *self-avoidance*: no two residues share a cell.

**Sequence.** A fixed string `sᵢ ∈ {H, P}`. Define `hᵢ = 1` if `sᵢ = H` else `0`.

**Energy (the objective).** With `ε = 1` (we measure energy in units of the
contact strength):

```
E = - Σ_{i<j, j≥i+2}  hᵢ hⱼ · [ |(xᵢ,yᵢ) − (xⱼ,yⱼ)| = 1 ]
```

The condition `j ≥ i+2` excludes chain neighbours (which are always adjacent and
do not count as a folding contact). `E` is a **non-positive integer**; the more
buried H–H pairs, the more negative.

**The sampling target.** MC draws conformations from the **Boltzmann
distribution** at temperature `T`:

```
P(state) ∝ exp(-E(state) / (k_B T))       (we absorb k_B into T, so T is in ε units)
```

**Metropolis–Hastings acceptance.** Propose a move from a state with energy
`E_old` to one with `E_new`, let `ΔE = E_new − E_old`. Accept the move with
probability

```
A = min(1, exp(-ΔE / T)).
```

Downhill or flat moves (`ΔE ≤ 0`) are always accepted; uphill moves are accepted
with a temperature-dependent chance, which is what lets the walk escape local
minima instead of getting stuck. Because `ΔE` is an **integer** here, `A` takes
only a handful of discrete values — a fact we exploit heavily (§5).

**Replica ladder (toward parallel tempering).** We run `R` independent walkers at
a geometric temperature ladder `T(r) = T_min · (T_max/T_min)^{r/(R-1)}`. Hot
replicas cross barriers; cold replicas refine minima. (Full parallel tempering
also *swaps* configurations between neighbouring temperatures — see §7.)

## 3. The algorithm

**One replica's walk** (`run_replica` in `src/mc_moves.h`):

1. Initialize the chain as a straight horizontal rod (always valid; deterministic).
2. Compute the starting energy `E = -contacts`.
3. Repeat `sweeps · n` times (one *sweep* = `n` attempts):
   a. Pick a random residue `i`.
   b. Propose a new lattice position for it (an end move if `i` is a terminus, a
      local corner/crankshaft move otherwise).
   c. Reject immediately if the move breaks connectivity or self-avoidance.
   d. Otherwise compute `ΔE`, draw `u ~ U[0,1)`, and **accept iff `u < A(ΔE)`**.
   e. Track the best (lowest) energy ever seen.
4. Return `{best_energy, final_energy}`.

**Complexity.** Let `n` = chain length, `S` = sweeps, `R` = replicas.
- Each attempt recomputes the energy in `O(n²)` (a naive all-pairs contact count;
  kept naive for clarity — a production code updates energy incrementally in
  `O(n)` or `O(1)` per move).
- One replica costs `O(S · n · n²) = O(S·n³)`.
- The full ensemble is `O(R · S · n³)` work.
- **Parallel structure:** the `R` replicas are *completely independent* — no
  shared writable state — so the parallel **depth** is just one replica's walk
  (`O(S·n³)`) and the **width** is `R`. That embarrassingly-parallel structure is
  exactly what the GPU exploits.

Arithmetic intensity is high relative to memory traffic: each thread keeps the
whole chain (`2n` integers) in registers/local memory and reads only a tiny
read-only Boltzmann table from global memory. This is a **compute-bound,
latency-tolerant** workload — ideal for the GPU's many-warp design.

## 4. The GPU mapping

**The pattern.** "Ensemble of independent histories" (PATTERNS.md §1): assign
**one GPU thread per replica**. This is the same shape as the ODE-ensemble
flagships (9.02, 13.02) and the per-history Monte-Carlo flagship (5.01).

**Thread-to-data map.** Thread `r = blockIdx.x · blockDim.x + threadIdx.x` runs
the entire walk for replica `r` and writes its one result to `out[r]`.

**Launch configuration.** `block = 256` threads (a multiple of the 32-lane warp,
enough warps to hide latency); `grid = ⌈R / 256⌉` blocks. The ragged last block
is guarded with `if (r >= n_replicas) return;`.

**Memory hierarchy and why:**

| Data | Where | Why |
|---|---|---|
| chain coords `x[],y[]` | per-thread **local memory** (register-backed for small `n`) | private to each walk; never shared → no synchronization |
| Boltzmann tables | **global memory**, read-only | small, read by each thread's own slice; read-only ⇒ no contention |
| `McProblem` (sequence, params) | kernel **parameter** space, by value | broadcast to all threads, read-only |
| `out[r]` | **global memory**, one writer | independent outputs ⇒ **no atomics** |

**Why no atomics (contrast project 5.01).** A Monte-Carlo *tally* (dose, k-means
centroids) has many threads adding into shared bins and needs `atomicAdd`. Here
each thread owns a private chain and a private output slot, so there is **zero
write contention** — the cleanest possible parallelization.

```
        replicas (one MC walker each)
   r=0     r=1     r=2     ...     r=R-1
  ┌────┐  ┌────┐  ┌────┐          ┌────┐
  │walk│  │walk│  │walk│   ...    │walk│     each thread:
  │ +  │  │ +  │  │ +  │          │ +  │       - private chain x[],y[]
  │RNG │  │RNG │  │RNG │          │RNG │       - own RNG stream (seed,r)
  └─┬──┘  └─┬──┘  └─┬──┘          └─┬──┘       - own Boltzmann table slice
    │       │       │               │
    ▼       ▼       ▼               ▼
  out[0]  out[1]  out[2]   ...   out[R-1]    one write each → no atomics

  blocks of 256 threads → grid = ceil(R/256)
```

**cuRAND vs our shared RNG (no black box).** Production GPU MC uses **cuRAND** to
generate device randomness. We deliberately *do not*: we use a tiny shared
**splitmix64** counter-based generator (in `mc_moves.h`) so the **CPU and GPU draw
bit-identical random numbers**, which makes verification exact (§6). cuRAND would
give each thread an independent high-quality stream (e.g. Philox) but would *not*
reproduce the CPU's sequence, so we would only be able to compare *distributions*,
not exact trajectories. Rolling our own also demystifies what cuRAND does: seed →
deterministic mixing → uniform draw.

## 5. Numerical considerations

**Precision.** Energies are **integers** (contact counts), so the entire
accept/reject decision chain is exact in that respect. The Boltzmann
probabilities are `double`.

**The determinism problem, and how we solve it.** The Metropolis test is a *hard
branch*: a single accept-vs-reject flip sends two trajectories to entirely
different conformations. If the host (MSVC) and device (nvcc) evaluated
`exp(-ΔE/T)` with even a 1-ULP difference, an accept could flip and the two walks
would diverge — making an exact CPU/GPU comparison impossible. We avoid this with
the key trick (PATTERNS.md §2, §4):

1. `ΔE` is a small **integer**, so `exp(-ΔE/T)` is one of only `2·R_de+1` discrete
   values.
2. We precompute those values **once on the host** (`build_boltzmann_table`) into
   a table, and **both** the CPU loop and the GPU kernel *look up the same table
   entry*. No transcendental runs inside the walk on either side.
3. The only floating-point operation in the hot loop is the comparison
   `u < table[ΔE]`, with identical `double` bits on both sides → identical accept
   decisions → **identical trajectories**.

The shared RNG closes the loop: replica `r` uses the same stream `(seed, r)` on
CPU and GPU, and **both branches of every attempt consume the same number of
random draws** (a rejected-by-geometry attempt still draws one uniform), so the
two stay in lockstep.

**Races / atomics.** None. Independent replicas, independent outputs, read-only
shared data. There is warp **divergence** (different replicas accept different
moves), which is intrinsic to MC; here the walks are short and balanced, so it
costs little.

## 6. How we verify correctness

`src/reference_cpu.cpp` runs the *identical* per-replica function the GPU runs
(`run_replica` from the shared header), just serially in a `for` loop. `main.cu`
runs both and compares **every replica's `{best_energy, final_energy}`**.

- **Tolerance: exactly 0.** Because the energies are integers produced by the same
  code with the same RNG and the same Boltzmann tables, the GPU result must equal
  the CPU result *bit for bit* (PATTERNS.md §4, the "exact" row). Any nonzero
  mismatch is a real bug, not floating-point noise. The demo prints
  `replica mismatches = 0`.
- **A second, scientific sanity check.** The reported `best energy = -8` means the
  ensemble found a fold with 8 buried H–H contacts starting from a straight chain
  — evidence that the move set is ergodic and the Metropolis weighting actually
  drives folding, not just that "CPU == GPU".
- **Edge cases handled by the loader:** `n` out of range, non-positive
  sweeps/replicas, `T_min ≤ 0` or `T_max < T_min`, a sequence whose length ≠ `n`,
  or a character other than `H`/`P` — each throws a clear error.

Agreement between an *independent* serial implementation and the parallel GPU one,
across all 256 replicas, is strong evidence the GPU code is correct.

## 7. Where this sits in the real world

This is a **reduced-scope teaching version**. Production protein MC differs in
almost every dimension:

- **Representation.** Real engines use full-atom or coarse-grained **3-D**
  structures with continuous backbone (`φ, ψ`) and side-chain (`χ`) dihedral
  angles — not a 2-D lattice. The conformational space is continuous and vast.
- **Energy function.** Instead of a contact count, they evaluate physics-based
  terms: **Lennard-Jones** van der Waals, electrostatics, hydrogen bonding,
  solvation, and statistical potentials (Rosetta's `ref2015`, AMBER force
  fields). This per-move energy evaluation is the expensive step the catalog
  notes should be GPU-accelerated.
- **Moves.** **Fragment insertion** (Rosetta swaps in backbone fragments from
  known structures), **rotamer packing** from the **Dunbrack** library for side
  chains, rigid-body domain moves, and loop closure — far richer than lattice
  corner flips.
- **Parallel tempering proper.** Real replica-exchange MC periodically **swaps**
  configurations between adjacent temperatures (accepting swaps by a Metropolis
  rule on `(βᵢ − βⱼ)(Eᵢ − Eⱼ)`). We run the temperature *ladder* but omit swaps;
  adding them is a natural exercise. Swaps introduce the one piece of inter-replica
  communication a GPU implementation must handle (a periodic, cheap sync).
- **RNG.** Production code uses **cuRAND** (Philox/XORWOW) for fast,
  high-quality independent device streams; exact CPU reproduction is sacrificed
  for speed and statistical quality.
- **Tools.** **Rosetta** (the reference MC design/folding suite, with experimental
  GPU extensions), **FoldX** (fast empirical energy for design), **OpenMM**
  (GPU MD/MC via custom integrators), and **ProteinMPNN** (learned sequence design
  that complements MC backbone sampling).

The lessons that *do* transfer directly: the Metropolis criterion, per-replica
RNG streams, the embarrassingly-parallel replica array, and the discipline of an
exact CPU reference.

---

## References

- **Lau, K.F. & Dill, K.A. (1989).** *A lattice statistical mechanics model of
  the conformational and sequence spaces of proteins.* Macromolecules 22:3986. —
  The HP model this project implements.
- **Metropolis et al. (1953); Hastings (1970).** The Metropolis–Hastings
  acceptance rule at the heart of the walk.
- **Rosetta** — <https://github.com/RosettaCommons/rosetta>: the canonical protein
  MC sampling/design engine; study its move set and score function.
- **OpenMM** — <https://github.com/openmm/openmm>: how GPU MC/MD is structured with
  custom integrators; a model for production device kernels.
- **FoldX** — <https://foldxsuite.crg.eu>: fast empirical energy evaluation for
  MC-based design.
- **ProteinMPNN** — <https://github.com/dauparas/ProteinMPNN>: GPU sequence design,
  complementary to backbone sampling.
- **Dunbrack rotamer library** — <https://dunbrack.fccc.edu/bbdep2010/>: the
  side-chain rotamers a full-atom version would pack.
