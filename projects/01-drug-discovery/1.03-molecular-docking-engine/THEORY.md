# THEORY — 1.3 Molecular Docking Engine

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use. This is a **reduced-scope teaching
> version**; §7 explains what production docking engines add._

---

## 1. The science

**Molecular docking** asks a question at the heart of computer-aided drug
discovery: *given a protein with a binding pocket and a small candidate molecule
(a "ligand"), how — and how well — does the ligand fit?* The answer has two parts:

1. the **binding pose** — the position, orientation, and (for flexible ligands)
   internal shape the ligand adopts inside the pocket; and
2. the **score** — a number estimating how favorable that binding is, a proxy for
   binding affinity. A more-negative score means a tighter predicted fit.

Why it matters: a pharma "virtual screen" may dock **millions of candidate
molecules** against one target protein to triage which few hundred are worth
synthesizing and testing in the lab. Docking is the computational filter that
makes that triage affordable.

The key structural fact that makes docking a *GPU* problem: scoring one pose is
**completely independent** of scoring any other pose. There are thousands of poses
per ligand and millions of ligands per campaign, and they can all be scored at
once. That is the massive data parallelism this project teaches.

This teaching version models a **rigid** ligand (no internal flexibility) against
a **single precomputed energy grid**, and searches an **exhaustive grid of poses**
(every translation × rotation on a lattice). Real engines relax all three of those
simplifications (§7), but the rigid-grid core is exactly the inner loop they spend
most of their time in.

## 2. The math

**The receptor as an energy grid.** Instead of recomputing protein–ligand
interactions atom-by-atom for every pose (expensive), docking engines *precompute*
the receptor's influence onto a regular 3-D lattice. Define a grid of points

$$ \mathbf{r}_{ijk} = \mathbf{o} + (i,j,k)\,h, \qquad i\in[0,n_x),\ j\in[0,n_y),\ k\in[0,n_z) $$

where $\mathbf{o}$ is the grid **origin** (Å) and $h$ the isotropic **spacing**
(Å). The stored value $G_{ijk}$ (kcal/mol) is the interaction energy a probe atom
would feel at $\mathbf{r}_{ijk}$. In a real force field this is a sum of a van der
Waals (Lennard-Jones 12-6) term and an electrostatic term; here the committed
sample uses one smooth Gaussian well (§6, `data/README.md`) so the answer is known.

**Sampling the grid at a continuous point.** A ligand atom lands at an arbitrary
world point $\mathbf{p}$, not on a lattice node, so we read the grid by
**trilinear interpolation** of the 8 surrounding corners. With fractional cell
coordinates $(f_x,f_y,f_z)\in[0,1)^3$ and corner energies $c_{abc}$
($a,b,c\in\{0,1\}$):

$$ E(\mathbf{p}) = \sum_{a,b,c\in\{0,1\}} c_{abc}\;
   w_a(f_x)\,w_b(f_y)\,w_c(f_z), \qquad w_0(f)=1-f,\ w_1(f)=f. $$

This is the continuous, once-differentiable energy surface every docking score
function is built on. (Implemented in `docking_core.h::trilinear_energy`.)

**A pose.** A rigid pose is a rotation $R$ then a translation $\mathbf{t}$ applied
to each ligand-local atom offset $\mathbf{l}_k$:

$$ \mathbf{p}_k = R\,\mathbf{l}_k + \mathbf{t}. $$

We parameterize $R = R_z(c)\,R_y(b)\,R_x(a)$ by three Euler angles $(a,b,c)$
(radians). Translation $\mathbf{t}=(t_x,t_y,t_z)$ is in Å.

**The score.** The pose energy is the weighted sum of grid energy over atoms:

$$ S(\text{pose}) = \sum_{k=1}^{N_\text{atoms}} w_k\, E(R\,\mathbf{l}_k + \mathbf{t}). $$

$w_k$ is a per-atom probe weight (think $|q_k|$, a partial-charge magnitude).
**Objective:** find the pose minimizing $S$ over the search space — the lowest
energy is the predicted binding mode. (Implemented in `docking_core.h::score_pose`.)

**Search space.** We sweep $n_\text{trans}$ translation samples per axis over
$[-R_\text{trans}, +R_\text{trans}]$ around a pocket centre, and $n_\text{rot}$
rotation samples per axis over $[0,2\pi)$. Total poses
$P = n_\text{trans}^3 \cdot n_\text{rot}^3$ (the sample: $7^3\cdot 3^3 = 9261$).

## 3. The algorithm

```
load receptor grid G, ligand L, search space S
best_energy = +inf ; best_index = 0
for p in [0, P):                      # P = n_trans^3 * n_rot^3 poses
    pose   = unrank_pose(S, p)        # decode flat index -> (t, a,b,c)
    energy = score_pose(G, L, pose)   # sum over atoms of trilinear grid energy
    if energy < best_energy:          # strict '<' => ties keep the LOWER index
        best_energy = energy ; best_index = p
report best_index, best_energy
```

**`unrank_pose` (the parity linchpin).** A single integer $p$ is decoded in
mixed-radix order into six sub-indices (3 translation, 3 rotation). This lets the
CPU loop iteration $p$ and the GPU thread with global index $p$ produce the
*identical* pose — essential for exact verification (§6).

**Complexity.**
- *Serial work:* $O(P \cdot N_\text{atoms})$ — each pose costs $N_\text{atoms}$
  atom transforms, each an 8-corner gather. The sample is $9261 \times 5$ scorings.
- *Parallel:* the $P$ pose scorings are independent, so the **work** is unchanged
  but the **depth** collapses to $O(N_\text{atoms})$ for the scoring plus
  $O(\log P)$ for the reduction to the best pose.
- *Arithmetic intensity:* each atom does ~8 grid reads + ~30 FLOPs. The trig for
  the rotation is hoisted out of the per-atom loop (computed once per pose), so it
  is amortized across all atoms — the single most important micro-optimization.
- *Access pattern:* the 8 grid corners are a small local gather; the four
  x-neighbours are contiguous (the grid is x-fastest), which is cache-friendly.

## 4. The GPU mapping

**Thread-to-data map.** *One thread per pose.* Thread with global index
`p = blockIdx.x*blockDim.x + threadIdx.x` scores pose `p`; a **grid-stride loop**
(`p += blockDim.x*gridDim.x`) lets a fixed, modest grid cover an arbitrarily large
pose count $P$. This is the canonical "independent jobs" pattern
(`docs/PATTERNS.md` §1, exemplar `1.12`).

**Launch configuration.** 256 threads/block (a multiple of the 32-lane warp, 8
warps to hide latency, good occupancy on `sm_75…sm_89`). Blocks =
`ceil(P / 256)`, capped at 4096 (the grid-stride loop covers any remainder).

**Memory hierarchy.**
- *Global memory:* the energy grid and ligand atoms (read-only, marked
  `__restrict__`). The grid is the bandwidth-relevant structure; the per-pose
  gather hits the L1/L2 cache because nearby poses read nearby grid cells.
- *Registers:* each thread keeps its running best `(energy, index)` packed key and
  the hoisted trig values — no spilling for this small kernel.
- *Shared memory:* a tiny array of one key per warp, used only in the block
  reduction (≤ 8 entries for 256 threads).
- *Texture memory (what production does):* AutoDock-GPU stores the grids in
  **texture memory**, whose hardware performs trilinear interpolation *for free*
  and caches spatially. We interpolate **by hand** in `trilinear_energy` so the
  math is visible and runs identically on the CPU — see §7.

**The reduction (the interesting CUDA bit).** We need not just the minimum energy
but *which* pose achieved it, **deterministically**. The trick (`kernels.cu`):
pack each result into one `unsigned long long`

```
key = ( order_preserving_uint32(float(energy)) << 32 ) | uint32(pose_index)
```

The high 32 bits are an **order-preserving** image of the energy (flip all bits if
negative, else flip the sign bit) so that *integer* `atomicMin` on the key
minimizes energy; the index in the low bits breaks ties toward the lower index —
**exactly** the CPU's `strict-<` rule. The block reduces with warp `__shfl_down_sync`
shuffles (no shared memory inside a warp) → one shared-memory step across warps →
one `atomicMin` per block into the global best. Integer `atomicMin` is associative
and commutative, so the result is **independent of thread/block ordering** →
byte-identical every run (`docs/PATTERNS.md` §3).

```
P poses ─▶ [thread per pose: score_pose -> pack_key]
                  │  (grid-stride loop over poses)
                  ▼
        warp min  (5x __shfl_down_sync)              ← no shared mem
                  ▼
        block min (per-warp keys via shared mem, 1 warp reduces)
                  ▼
        atomicMin(d_best, block_min)                 ← 1 atomic / block
                  ▼
        host: decode index, recompute EXACT double energy
```

## 5. Numerical considerations

- **Precision.** All scoring math is `double` (FP64). Docking energies span a wide
  dynamic range and we want the CPU and GPU to agree to many digits, so we avoid
  FP32 in the physics. The *reduction key* uses the FP32 image of the energy only
  to fit `(energy,index)` in 64 bits for the tie-break — and then the host
  **recomputes the winner's exact FP64 energy** from its index, so no precision is
  lost in the reported number.
- **Determinism.** The headline result (best pose index + energy) is **bit-for-bit
  reproducible**: the reduction is an *integer* `atomicMin`, which (unlike a float
  `atomicAdd`) does not depend on the nondeterministic order in which blocks
  finish. This is the central lesson of `docs/PATTERNS.md` §3 applied to a *min*
  rather than a *sum*.
- **Race conditions.** None on the scoring (each thread writes only its own
  register). The only shared write is `atomicMin(d_best, …)`, which is atomic by
  construction.
- **Tie handling.** Putting the index in the low bits of the key makes "lowest
  energy, then lowest index" a single unsigned comparison — so even genuinely
  tied energies pick the same winner on both CPU and GPU.
- **`__CUDACC__` constant parity.** The `2π` used to lay out the rotation sweep is
  a hard literal in *both* `reference_cpu.cpp` (`TWO_PI`) and `kernels.cu`
  (`D_TWO_PI`), so the host and device enumerate the identical poses (MSVC does not
  define `M_PI` by default — a literal sidesteps that portability trap).

## 6. How we verify correctness

The CPU reference (`src/reference_cpu.cpp::dock_cpu`) is a single flat loop with no
parallelism — obviously correct by inspection. Crucially, **both** the CPU and the
GPU call the *same* `score_pose()` and `unrank_pose()` (the shared
`__host__ __device__` core in `docking_core.h`, `docs/PATTERNS.md` §2), so they
execute byte-identical arithmetic.

The check in `main.cu` is therefore strong and two-layered:
1. **Same winner (exact).** The GPU's best pose **index** must equal the CPU's —
   an exact integer match. Because the reduction is deterministic and ties break
   identically, this holds exactly.
2. **Same energy (tolerance `1e-9`).** Both recompute the winner's energy with the
   same `score_pose`, so they agree to round-off; `1e-9` kcal/mol is a generous
   margin that also guards the degenerate "different index, equal energy" tie.

On the committed sample the observed `energy_err` is **exactly 0** and the indices
match (`CPU idx = GPU idx = 4967`). A second, *physical* check validates the
science, not just CPU==GPU agreement: the recovered best translation
**(0.5, −0.5, 0.0) Å** equals the synthetic well's location by construction
(`data/README.md`) — the ligand centroid drops into the pocket, as it must.

Edge cases handled: atoms that drift to/just past the grid boundary read the **edge
value** (`clampi`, a "wall" that penalizes out-of-pocket poses); $n=1$ along an
axis collapses to the midpoint/zero angle; an all-flat grid scores every pose equal
and the tie rule then deterministically returns index 0.

## 7. Where this sits in the real world

This is a **reduced-scope teaching version**. Production engines (catalog "Prior
art") keep the rigid-grid scoring inner loop but add:

- **Per-atom-type grids + electrostatics.** Real AutoGrid precomputes one energy
  map *per ligand atom type* (C, N, O, …) plus a separate electrostatics map and a
  desolvation map; the score indexes the *right* map per atom. We use one generic
  grid. The trilinear gather is identical — only the number of maps changes.
- **Texture memory.** AutoDock-GPU binds those maps to CUDA **texture** objects so
  the hardware does the trilinear interpolation and spatial caching. Swapping our
  hand-rolled `trilinear_energy` for a `tex3D<float>` fetch is a natural exercise
  (and the catalog's named GPU pattern).
- **Smart search instead of brute force.** Exhaustive pose enumeration does not
  scale to flexible ligands. AutoDock-GPU runs a **Lamarckian Genetic Algorithm**
  (a GA whose individuals are refined by **BFGS local search** — "Lamarckian"
  because the learned improvement is written back into the genome). Vina-GPU 2.1
  uses Monte-Carlo + a randomized iterated local search with BFGS. One GA
  individual per thread-block, warp-reduced fitness — the same parallel skeleton
  as here, with a cleverer pose generator than our lattice.
- **Torsional flexibility.** Real ligands rotate about rotatable bonds; a pose is
  then translation + orientation + a torsion vector, scored after rebuilding the
  conformation. We hold the ligand rigid.
- **Quaternions, not Euler angles.** To avoid gimbal lock and sample orientations
  uniformly, production code uses quaternions/axis-angle. Our Euler triple is the
  most readable parameterization for a first look.
- **Learned scoring.** GNINA replaces the hand-built force field with a 3-D CNN
  scorer. The pose-sampling parallelism is unchanged; the *scorer* becomes a neural
  net.

Reported speedups are large because the per-pose score is cheap and embarrassingly
parallel: AutoDock-GPU cites >1000× over single-CPU AutoDock4, Uni-Dock >2000× on a
V100. On our toy 9261-pose sample the GPU is launch/copy-bound (a teaching artifact,
not a benchmark, CLAUDE.md §12) — the advantage appears once $P$ and the ligand
count are realistic.

---

## References

- **AutoDock-GPU** — <https://github.com/ccsb-scripps/AutoDock-GPU>. The canonical
  CUDA/OpenCL docking engine; study its LGA + grid-texture design — the direct
  ancestor of this project's pattern.
- **Uni-Dock** — <https://github.com/dptech-corp/Uni-Dock>. GPU batch docking;
  learn how throughput is reached by docking many ligands per launch.
- **Vina-GPU 2.1** — <https://github.com/DeltaGroupNJUPT/Vina-GPU-2.1>. GPU AutoDock
  Vina with RILC-BFGS; learn the Monte-Carlo + local-search search strategy.
- **GNINA** — <https://github.com/gnina/gnina>. CNN-scored docking; learn how a
  learned scorer slots into the same sampling loop.
- **Trott & Olson (2010), *AutoDock Vina*, J. Comput. Chem. 31:455** — the modern
  empirical scoring function and search; the reference for "what a real score is".
- **Morris et al. (2009), *AutoDock4 / AutoDockTools*, J. Comput. Chem. 30:2785** —
  the grid-based force field and the Lamarckian GA this project abstracts.
