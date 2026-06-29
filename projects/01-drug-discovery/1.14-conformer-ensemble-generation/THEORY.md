# THEORY — 1.14 Conformer Ensemble Generation

> The deep didactic explanation (the "why"). Written for a sharp student who knows C++ but is new to CUDA and
> new to this domain. See [README.md](README.md) for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

A molecule is not a rigid object. Single (σ) bonds rotate almost freely, so a flexible molecule constantly
interconverts between many 3D shapes called **conformers** (or conformations). They differ *only* in their
**torsion angles** — the dihedral angle around each rotatable bond — because bond *lengths* and bond *angles*
are very stiff and barely move at room temperature. The set of low-energy conformers a molecule actually
populates is its **conformer ensemble**.

Why does this matter for drug discovery? Almost every 3D method assumes you already have the right shape:

- **Docking** fits a ligand into a protein pocket. The "bioactive conformer" the protein binds is usually
  *not* the global minimum in solution — so you must supply a *diverse ensemble* and let docking pick.
- **3D / shape / pharmacophore similarity search** compares molecular shapes; you need each molecule's
  realistic shapes first.
- **Free-energy and QSAR models** average over the populated conformers.

So "generate the conformer ensemble" is a universal *preprocessing* step. For a library of millions of
molecules, each needing thousands of conformers, it is a serious compute bottleneck — and an
embarrassingly parallel one, which is exactly why it belongs on a GPU.

This project teaches the full pipeline — **enumerate → embed → score → prune** — on a deliberately small
molecule: an unbranched chain of `N_ATOMS = 8` atoms (think the carbon backbone of n-octane). It has
`N_TORSION = N_ATOMS − 3 = 5` rotatable torsions. We allow each torsion three classic staggered values
(**anti** 180°, **gauche+** +60°, **gauche−** −60°), the three minima of a saturated bond's torsion
potential. That gives `3⁵ = 243` conformers to generate and score.

## 2. The math

**Internal coordinates.** A conformer of our chain is fully described by the torsion vector
**φ** = (φ₁, …, φ₅), φₜ ∈ {180°, +60°, −60°}, with bond length `L = 1.53 Å` and bond angle `θ = 111°` fixed.

**Embedding (internal → Cartesian).** To score a conformer we need 3D atom positions **rᵢ ∈ ℝ³**. Atom `i`
(for `i ≥ 3`) is placed from its three predecessors `A=i−3, B=i−2, C=i−1` using the **Natural Extension
Reference Frame (NeRF)** construction. Build an orthonormal frame at `C`:

```
bc = unit(C − B)                 # along the bond we extend from
n  = unit( (B − A) × bc )        # plane normal — the dihedral axis
m  = n × bc                      # completes a right-handed frame {bc, m, n}
```

and place the new atom at

```
rᵢ = C + L·( −cosθ · bc  +  sinθ·cosφ · m  +  sinθ·sinφ · n )
```

This is exact: it reproduces the requested bond length `L` to `C`, the requested bond angle `θ` at `C`, and
the requested dihedral `φ` about the `B→C` axis. The first three atoms have no torsional freedom and are
pinned to a fixed seed geometry (origin, +x, xy-plane).

**Energy (force field).** Each conformer gets a scalar potential energy `E(φ)` in kcal/mol, the sum of two
classic terms:

- **Torsion term** (bonded): a 3-fold periodic potential per rotatable bond,
  `E_tors = Σₜ ½·V₃·(1 + cos 3φₜ)`, with `V₃ = 2 kcal/mol`. It is minimized at `φ = 180°` (anti), since
  `cos(3·180°) = cos 540° = −1`. This is why the **all-anti** conformer (index 0) is the torsional minimum.
- **Nonbonded term** (steric clash): a repulsive Lennard-Jones wall over atom pairs separated by more than two
  bonds, `E_nb = Σ_{i, j≥i+3} ε·(σ/r_{ij})¹²`, with `ε = 0.10 kcal/mol`, `σ = 3.40 Å`. It penalizes conformers
  that fold back on themselves. To keep it numerically sane we **soft-core** it: floor `r_{ij}` at
  `R_min = 1.5 Å` so the wall saturates instead of diverging (see §5).

`E(φ) = E_tors(φ) + E_nb(r(φ))`. **Inputs:** the conformer index (→ φ via mixed-radix decode). **Output:** the
scalar `E`. **Objective of the project:** compute `E` for all 243 conformers, then select a non-redundant
low-energy subset.

**Pruning (RMSD clustering).** Two conformers with nearly identical shape are redundant. We measure shape
distance by **root-mean-square deviation** of atom positions,
`RMSD(a,b) = sqrt( (1/N)·Σᵢ |rᵢ(a) − rᵢ(b)|² )`. Because every conformer shares the same first three (seed)
atoms in the same lab frame, no superposition is needed — a simplification we exploit (the Kabsch version is
an exercise). Greedy "leader" clustering then keeps a conformer as a representative only if its RMSD to every
already-kept representative exceeds a threshold (1.0 Å here, a typical RDKit value).

## 3. The algorithm

```
enumerate_and_score:                          # the parallel part (GPU + CPU reference)
  for each conformer c in 0 .. 3^5 - 1:
      φ   = decode_torsions(c)                 # mixed-radix: digit t -> rotamer of torsion t
      r   = build_coords(φ)                    # NeRF embedding, O(N_ATOMS)
      E[c] = torsion_energy(φ) + nonbonded_energy(r)   # O(N_ATOMS^2) pairwise clash

prune (RMSD clustering):                       # the serial part (CPU)
  order = argsort(E)                           # ascending energy
  reps = []
  for c in order:
      if min_k RMSD(coords(c), coords(reps[k])) >= threshold:
          reps.append(c)
```

**Complexity.** Per conformer the work is `O(N_ATOMS²)` (dominated by the pairwise clash sum). The full sweep
is `O(N_CONFORMER · N_ATOMS²)`. Every conformer is **independent** — its energy depends only on its own index
— so the *parallel depth* is just `O(N_ATOMS²)` (one conformer per thread, all in flight at once); the
*parallel work* equals the serial work. The pruning is `O(reps · N_CONFORMER · N_ATOMS)` and is inherently
**sequential** (each acceptance decision depends on the ones before it), so it stays on the CPU.

**Arithmetic intensity & access pattern.** Each thread reads *nothing* from global memory (a conformer is
described entirely by its integer thread index) and writes exactly *one* double. All intermediate coordinates
live in registers/local memory. So this kernel is compute-bound on the trig (`cos/sin/sqrt`) of the embedding
and the pairwise multiplies of the clash term — there is essentially no memory traffic to optimize, which is
unusual and worth noticing.

## 4. The GPU mapping

**One thread per conformer.** Thread `(blockIdx.x, threadIdx.x)` owns conformer
`c = blockIdx.x·blockDim.x + threadIdx.x`, with a grid-stride loop so a fixed-size grid covers any
`N_CONFORMER`. The thread decodes φ, builds the 8 atom positions in a tiny stack array (registers / local
memory), sums the energy, and writes `out[c]`.

```
 conformers:   0    1    2    3   ...                          242
               |    |    |    |                                 |
 threads:     t0   t1   t2   t3  ...   (grid-stride for c >= #threads)
               \    \    \    \                                 /
                +--- each thread: decode φ -> NeRF embed -> sum E -> store out[c]
 global mem:   [ out[0] out[1] out[2] ... out[242] ]   <- the ONLY global writes (coalesced)
```

- **Launch config:** `block = 128` threads (a multiple of the 32-lane warp; small enough to keep register
  pressure low given each thread's `N_ATOMS`-sized scratch, large enough to give the scheduler several warps
  to hide the latency of the transcendental functions). `grid = ceil(N/128)`, capped at 1024 with the
  grid-stride loop covering the rest.
- **Memory hierarchy:** **registers/local** for the per-conformer coordinates and energy; **global** only for
  the one output store per thread, which is *coalesced* because consecutive threads write consecutive indices.
  No **shared** memory and no **constant** memory are needed — there is no data shared between threads. No
  **atomics** — outputs are disjoint.
- **Occupancy:** limited by registers, not by shared memory or block count. Because the work set is tiny and
  fixed, occupancy is high; the kernel is latency-hidden by having many warps resident.
- **Libraries:** none here — the kernel is a few dozen lines of plain device math. The catalog mentions
  cuSOLVER (batched distance-geometry SVD) and a custom pairwise-RMSD kernel for the *full* problem; see §7
  and the exercises. We deliberately avoid a library so nothing is a black box at this scale.

## 5. Numerical considerations

- **Precision: FP64 throughout.** The embedding chains `cos/sin/sqrt` and the clash term raises ratios to the
  12th power; in `float` the rounding would accumulate visibly. We use `double` so the GPU and CPU agree to
  ~`1e-12` (exercise 5 asks you to measure the FP32 degradation).
- **The soft core is a numerical necessity, not just chemistry.** A bare `(σ/r)¹²` wall explodes to ~`1e14`
  kcal/mol when a folded conformer brings two atoms nearly on top of each other. At that magnitude the *last
  bit* of difference between the host `libm` and the device's fused-multiply-add (FMA) is amplified to whole
  kcal/mol — so CPU and GPU would disagree by ~4 kcal/mol for those conformers and exact verification would be
  impossible. Flooring `r²` at `R_min²` saturates the wall to a finite, well-conditioned value. This mirrors
  the **soft-core potentials** used in real free-energy calculations, and it is the single most important
  numerical lesson in this project.
- **Determinism.** There are **no floating-point reductions** across threads — each thread writes its own
  independent result — so there is no atomics-reordering nondeterminism (contrast §3 of PATTERNS.md). The
  printed result is therefore byte-identical every run (verified). The clustering uses a `stable_sort` with a
  lower-index tie-break, so ties in energy resolve deterministically too.
- **FMA caveat.** The GPU may fuse `a*b + c` into a single rounded operation where the CPU rounds twice; that
  is the source of the residual ~`1e-12` difference. It is real, expected, and physically negligible.

## 6. How we verify correctness

Two independent checks:

1. **GPU vs CPU agreement.** `src/reference_cpu.cpp` computes all 243 energies with an obvious serial loop;
   `src/kernels.cu` computes them on the GPU. Both call the *same* `conformer_energy()` from
   `src/conformer.h` (the shared `__host__ __device__` core), so they run identical math and must agree. We
   require `max|E_cpu − E_gpu| ≤ 1e-9` kcal/mol; the observed error is ~`5e-12` (pure FMA rounding,
   PATTERNS.md §4 "short double-precision"). Agreement between an independent serial implementation and the
   parallel one is strong evidence both are correct — a bug would have to corrupt *both* identically.
2. **A known-answer sanity check.** The **global minimum is the all-anti conformer** (torsions
   `+180 +180 +180 +180 +180`), which is the chemically correct lowest-torsion-energy, lowest-clash shape of a
   saturated chain. If the embedding or the force field were wrong, this would not come out right.

Edge cases handled: degenerate near-overlap (soft core), the all-zero direction in `vunit` (guarded),
ragged last block on the GPU (grid-stride bound), and an empty/short sample file (falls back to defaults).

## 7. Where this sits in the real world

Production conformer generation differs from this teaching version in scope, not in spirit:

- **RDKit ETKDG (ETKDGv3).** Works on an *arbitrary* molecular graph. It does **distance geometry**: build a
  distance-bounds matrix from the 2D graph *and from experimental torsion-angle preferences* (the "ET" =
  Experimental Torsion Knowledge), then **embed** random coordinate sets that satisfy those bounds (an
  eigendecomposition / SVD of a metric matrix — the batched-SVD step the catalog assigns to **cuSOLVER**),
  and finally **minimize** a full **MMFF94** force field. We replace the stochastic distance-geometry guess
  with a deterministic enumeration of discrete rotamers, and the 80-term MMFF94 with two terms — but the
  embed-then-score-then-prune loop is exactly the same.
- **Scale & the GPU win.** Real runs generate thousands of conformers for each of millions of molecules. That
  is where one-thread-per-conformer (or one-block-per-molecule) pays off massively; at our 243-conformer toy
  scale the GPU is *slower* (launch/copy overhead dominates — see the timing caveat).
- **Pairwise RMSD at scale.** Deduplicating thousands of conformers needs an N×N RMSD matrix — a second,
  separately parallelizable kernel (one thread per pair, the catalog's "custom CUDA kernels for pairwise
  RMSD"; exercise 1).
- **ML conformer generators.** TorsionalDiffusion and GeoMol learn the torsion distribution from
  crystallographic data and *sample* conformers with a GPU neural network instead of enumerating them — the
  current research frontier, operating on the very same torsional degrees of freedom we enumerate by hand.

---

## References

- **RDKit ETKDG** — Riniker & Landrum, *J. Chem. Inf. Model.* 2015; ETKDGv3: Wang, Witek, Landrum, Riniker
  2020. The canonical open-source method; study the distance-geometry embed + MMFF refine pipeline.
  https://github.com/rdkit/rdkit
- **NeRF construction** — Parsons et al., "Practical conversion from torsion space to Cartesian space for in
  silico protein synthesis," *J. Comput. Chem.* 2005. The internal→Cartesian recipe in `build_coords`.
- **MMFF94** — Halgren, *J. Comput. Chem.* 1996. The force field our torsion + nonbonded terms abbreviate.
- **GEOM** — Axelrod & Gómez-Bombarelli 2022; 37M DFT conformers. The real dataset to validate against.
  https://github.com/learningmatter-mit/geom
- **TorsionalDiffusion** — Jing et al., NeurIPS 2022. https://github.com/gcorso/torsional-diffusion
- **GeoMol** — Ganea et al., NeurIPS 2021. https://github.com/PattanaikL/GeoMol
- **Soft-core potentials** — Beutler et al., *Chem. Phys. Lett.* 1994. Why flooring the repulsion is standard
  practice in molecular simulation.
