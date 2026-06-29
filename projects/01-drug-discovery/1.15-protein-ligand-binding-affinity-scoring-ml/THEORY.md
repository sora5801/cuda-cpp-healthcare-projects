# THEORY — 1.15 Protein-Ligand Binding Affinity Scoring (ML)

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

A drug works by **binding** a target protein — slotting into a pocket on its
surface and sticking there long enough to switch the protein's behavior. How
*tightly* it sticks is the **binding affinity**, summarized by the dissociation
constant `Kd` (the ligand concentration at which half the protein is occupied).
Chemists prefer the log scale:

```
pKd = -log10(Kd)
```

so `Kd = 1 nM` (a strong binder) is `pKd = 9`, and `Kd = 1 mM` (a weak one) is
`pKd = 3`. A virtual screen docks millions of candidate molecules into the pocket
and must **rank** them by predicted affinity; only the top few are ever
synthesized and tested. The function that turns a docked 3D structure into a number
is the **scoring function**, and it is the rate-limiting accuracy bottleneck of the
whole pipeline.

Two families of scoring functions exist:

- **Physics-based** (force fields, MM/GBSA, free-energy perturbation): compute an
  interaction energy from first principles. Accurate but slow — FEP can take a
  GPU-hour *per pose*.
- **Machine-learned**: *learn* the structure → affinity map from a database of
  complexes whose `Kd` was measured experimentally (PDBbind has ~19,000). The model
  never simulates physics; it pattern-matches. Inference is ~`1 ms`/pose, so an ML
  scorer can rescore an entire docking run.

The dominant ML architecture for this is the **3D convolutional neural network**:
treat the protein-ligand complex like a tiny 3D image (a "molecular MRI") and run
an image CNN over it. That is exactly what we implement — a small, explicit,
verifiable 3D-CNN forward pass. (GNINA is the production embodiment of this idea.)

> **Honesty up front:** our network is **untrained** (weights from a deterministic
> generator) and our data is **synthetic**. We are teaching the *GPU inference
> pattern* of a 3D-CNN scorer, not making affinity predictions. See §7.

## 2. The math

**Input.** A docked pose is a set of atoms `{(rᵢ, eᵢ, sᵢ)}`, where `rᵢ ∈ ℝ³` is an
atom's position (angstroms), `eᵢ ∈ {C,N,O,S}` its element, and `sᵢ ∈ {protein,
ligand}` its side. **Output.** One scalar `pKd ∈ [2, 11]`.

**Stage 1 — Voxelize.** Place a `G×G×G` grid (`G = 16`, spacing `δ = 1 Å`, box
`16 Å`) over the pocket. Each atom is smeared into a Gaussian density blob. The
density in channel `c` at voxel center `v` is

```
ρ_c(v) = Σ_{i : channel(eᵢ,sᵢ)=c}  exp( -‖v - rᵢ‖² / (2σ²) ),   σ = 0.8 Å
```

with a hard cutoff at `‖v - rᵢ‖ > 3 Å` (beyond which the Gaussian is < `10⁻³`).
The channel index is `channel(e,s) = s·4 + e`, giving `C_in = 4 elements × 2 sides
= 8` channels. So the grid is a tensor `ρ ∈ ℝ^{8 × 16 × 16 × 16}`.

**Stage 2 — 3D convolution + ReLU.** `C_out = 8` filters, each `W^{(o)} ∈ ℝ^{8 ×
3 × 3 × 3}`, slide over the grid with SAME padding:

```
a_o(x,y,z) = Σ_{c=0}^{7} Σ_{dz,dy,dx ∈ {-1,0,1}}  W^{(o)}_{c,dz,dy,dx} · ρ_c(x+dx, y+dy, z+dz)
h_o(x,y,z) = ReLU( a_o(x,y,z) ) = max(0, a_o(x,y,z))
```

(out-of-grid taps contribute 0). This yields 8 rectified feature maps
`h ∈ ℝ^{8 × 16³}`.

**Stage 3 — Global average pool.** Collapse each map to one number:

```
p_o = (1 / 16³) Σ_{x,y,z} h_o(x,y,z)        →   p ∈ ℝ⁸
```

**Stage 4 — Dense readout + squash.** A linear layer to a scalar, then a logistic
squash into the affinity window:

```
raw      = b + Σ_{o=0}^{7} u_o · p_o
pKd      = 2 + 9 · σ_logistic(raw),     σ_logistic(t) = 1 / (1 + e^{-t})
```

The weights `W, u, b` are, in a real scorer, *learned* by minimizing
`Σ (pKd_pred − pKd_measured)²` over PDBbind. Here they are fixed pseudo-random
numbers `lcg_weight(index) ∈ [-1, 1)` (a deterministic LCG hash) — see §7.

## 3. The algorithm

For each complex, run stages 1→4. Pseudocode:

```
for each complex:
    zero grid[8][16][16][16]
    for each atom a:                          # voxelize (scatter or gather)
        for each voxel v within 3 Å of a:
            grid[channel(a)][v] += gauss(|v - r_a|)
    for each output map o, voxel (x,y,z):     # conv + ReLU
        acc = Σ_c Σ_{3³ nbhd} W[o,c,..] * grid[c, nbhd]
        pooled[o] += relu(acc)
    pooled[o] /= 16³                          # global average
    raw = b + Σ_o u[o]*pooled[o]              # dense
    pKd = 2 + 9*logistic(raw)
```

**Complexity (one complex).** Voxelization is `O(A · 7³)` for `A` atoms (each atom
touches a `~7³` voxel box). The conv dominates: `O(C_out · G³ · C_in · K³) = 8 ·
4096 · 8 · 27 ≈ 7.1 M` multiply-adds. Pool and dense are negligible (`O(C_out·G³)`
and `O(C_out)`). For a batch of `N` complexes the serial cost is `N ×` that — and
crucially **every complex is independent**, so the *parallel depth* collapses: with
enough hardware all `N` poses finish in the time of one.

**Arithmetic intensity.** The conv reads each of the `8 · 16³` grid values `~27 ×
C_out` times. A naive implementation is therefore **memory-bandwidth-bound** unless
those reads hit cache or shared memory — the classic stencil optimization story
(see exercise 3 and PATTERNS.md §1).

## 4. The GPU mapping

Two stacked parallel patterns (PATTERNS.md §1):

- **Batch over complexes → one thread BLOCK per complex.** `blockIdx.x` selects the
  pose. This is the "independent jobs" pattern and mirrors the real workload
  (rescore millions of independent poses).
- **Stencil within a complex → threads cooperate over voxels.** `threadIdx.x` is a
  worker that strides over the `16³ = 4096` voxels.

```
        grid of N blocks                     one block (BLOCK=128 threads)
   ┌──────┬──────┬─────┬──────┐         ┌───────────────────────────────┐
   │ blk0 │ blk1 │ ... │ blkN │         │ pass 1: voxelize (gather)     │
   │ pose │ pose │     │ pose │  ──►     │   thread t owns voxels        │
   └──────┴──────┴─────┴──────┘         │     t, t+128, t+256, ...      │
   each block scores ONE pose           │ __syncthreads()               │
                                        │ pass 2: conv+ReLU, per-thread │
   d_grids: one 8×16³ grid slice        │   partial pooled[8] sums      │
   per block, in global memory          │ tree-reduce pooled in shared  │
                                        │ thread 0: dense + squash      │
                                        └───────────────────────────────┘
```

**Pass 1 — voxelize as a GATHER, not a scatter.** The intuitive mapping (one thread
per atom, *scatter* its blob into voxels) needs **atomic** adds, and atomic
`double` adds sum in a nondeterministic order → irreproducible output
(PATTERNS.md §3). We invert it: each thread owns a set of **voxels** and *gathers*
the contributions of all nearby atoms. Now every voxel has exactly one writer →
**race-free, deterministic, and order-matched to the CPU**.

**Pass 2 — conv + ReLU + pooling.** Each thread sweeps its strided voxels, computes
all 8 conv responses per voxel (reading the `3³` neighborhood from the grid in
global memory), ReLUs them, and accumulates into its own `partial[8]` registers.
Then the block combines the per-thread partials with a **shared-memory halving-tree
reduction** (`red[t] += red[t+stride]`), which has a fixed, deterministic addition
order. Thread 0 finishes the dense layer and writes `out[blockIdx.x]`.

**Launch config.** `<<<N, 128>>>`. `BLOCK = 128` is a multiple of the 32-lane warp
(4 warps), enough to hide global-memory latency while keeping the pooling reduction
and shared footprint small (`128 doubles = 1 KB`). With `4096` voxels that is `~32`
voxels/thread — a healthy amount of work to amortize the launch.

**Memory hierarchy.** The density grid lives in **global memory** (one `256 KB`
slice per block; too big for shared memory). The conv weights are regenerated on
the fly from `lcg_weight()` — pure ALU, no memory traffic, and identical on CPU and
GPU (the whole point of the shared `scoring_core.h`). Per-thread `partial[8]` lives
in **registers**; the pooling reduction uses **shared memory** (`red[128]` +
`pooled[8]`). A production kernel would additionally tile the grid neighborhood into
shared memory to cut the `~27×` redundant global reads (exercise 3).

**No CUDA library is used here on purpose.** A production 3D-CNN calls **cuDNN**'s
`cudnnConvolutionForward` (which internally picks an implicit-GEMM, FFT, or Winograd
algorithm). We hand-roll the conv so there is **no black box** (CLAUDE.md §6.1.6):
the loop you read *is* the convolution cuDNN computes, just unfused and in double
precision. §7 says what cuDNN adds.

## 5. Numerical considerations

- **Precision: FP64 (double) everywhere.** A real scorer runs FP16/FP32 for speed;
  we use `double` so the CPU and GPU forward passes can be compared at near
  machine precision and the verification is convincing. (Exercise 4 switches to
  FP16 and explores the accuracy/speed trade.)
- **Determinism.** The only non-associative step is the global-average **pool sum**
  over 4096 voxels. The CPU sums them left-to-right; the GPU sums them with a
  binary tree. Both orders are *fixed* (so each side is reproducible run-to-run),
  but they differ from each other by a few ULPs (`~10⁻¹²` relative). We deliberately
  avoided `atomicAdd` on doubles precisely to keep both sides deterministic
  (PATTERNS.md §3). Voxelization is exactly order-matched (gather sums atoms in
  index order on both sides), and the Gaussian uses a hard cutoff so both sides sum
  the *identical set* of terms.
- **Stability.** The logistic squash keeps the output bounded in `[2, 11]`
  regardless of `raw`, so there is no overflow path. The Gaussian `exp` argument is
  always `≤ 0`, so `exp` is in `(0, 1]` — no `inf`.
- **Race conditions.** None. Gather voxelization has one writer per voxel; the
  pooling reduction is guarded by `__syncthreads()` at every tree level.

## 6. How we verify correctness

`src/reference_cpu.cpp::score_cpu()` is an independent, single-threaded forward pass
that calls the **same** per-element functions from `scoring_core.h` as the kernel
(the HD-macro idiom, PATTERNS.md §2). `main.cu` runs both and computes
`max_abs_err = max_i |pKd_cpu[i] − pKd_gpu[i]|`.

**Tolerance: `1e-6` pKd.** Justification (PATTERNS.md §4): the two implementations
run identical double-precision math and differ *only* in the pooling reduction
order, which perturbs each pooled feature by `~10⁻¹²`. Propagated through the linear
dense layer and the smooth logistic squash, that stays `~10⁻¹²` in pKd. We verify to
`1e-6` — six orders of magnitude of headroom above the real disagreement, and two
orders *below* the `1e-4` precision we print, so a genuine bug (wrong index, wrong
weight, missing ReLU) would blow past it immediately. On the reference machine the
observed error was `~2e-15`.

**Why this is convincing.** The CPU and GPU codepaths share only the scalar math;
the *loop structure, memory layout, parallel reduction, and launch* are completely
different. If a transcription error existed in either the voxel indexing, the
`3³ × 8 × 8` conv index arithmetic, or the weight-index formula, the two would
disagree by far more than `1e-6`. Agreement across two such different
implementations is strong evidence both are right. **Edge cases** exercised by the
sample: ligands of different sizes, atoms near the box edge (SAME-padding taps that
read outside the grid → 0), and channels with no atoms (all-zero density → pooled 0).

A *stronger* check a real project adds — and we cannot, because the model is
untrained — is correlation against measured `pKd` on a held-out set (CASF-2016
reports Pearson `R ≈ 0.8` for good scorers). Here the "label" column is synthetic
and the network never saw it, so we make **no** accuracy claim.

## 7. Where this sits in the real world

What a production 3D-CNN scorer (e.g. **GNINA**) does that this teaching version
omits:

- **Trained weights.** GNINA's CNN is fit on tens of thousands of complexes with
  pose-labeled good/bad examples. Our `lcg_weight()` is a deterministic stand-in so
  the demo is self-contained; swapping in loaded weights is exercise 5.
- **cuDNN, not a hand loop.** Real layers call `cudnnConvolutionForward`, which
  chooses implicit-GEMM / FFT / Winograd algorithms and fuses bias+activation. Our
  explicit loop computes the *same* convolution so you can see it; cuDNN is ~100×
  faster and FP16-capable.
- **Mixed precision + batching at scale.** FP16 tensor-core math and large batched
  launches (thousands of poses per kernel, streamed through a fixed grid-buffer
  pool) are how you actually hit `~1 ms`/pose. Our one-grid-per-pose allocation is
  the clearest layout, not the scalable one (see the note in `score_gpu`).
- **Richer features.** Production grids encode partial charge, hybridization,
  aromaticity, H-bond donor/acceptor flags — not just element. And the box is
  centered and oriented on the detected pocket.
- **Architectural alternatives.** Equivariant **GNNs** (SchNet, DimeNet++, via
  PyTorch Geometric) operate on the atom graph directly and are rotation-invariant
  by construction — no voxel-grid orientation problem. **Co-folding** transformers
  (NeuralPLexer) predict the bound structure *and* affinity jointly. The catalog
  lists all of these as the frontier.

This project is a deliberately **reduced-scope teaching version** (CLAUDE.md §13):
the smallest thing that is still recognizably a 3D-CNN affinity scorer and that runs
the real GPU pattern end-to-end, exactly verifiable.

---

## References

- **GNINA** — McNutt et al., *J. Cheminformatics* 2021; <https://github.com/gnina/gnina>.
  The reference CNN-rescoring tool; read it for the atom-gridding and the trained
  architecture this project miniaturizes.
- **DeepChem** `AtomicConvolutions` / MPNN — <https://github.com/deepchem/deepchem>.
  The graph-based alternative to voxel grids; contrast with our CNN.
- **PDBbind** — <http://www.pdbbind.org.cn>; **CASF-2016** —
  <http://www.pdbbind.org.cn/casf.php>. The data and benchmark a real scorer is
  trained and evaluated on.
- **NeuralPLexer** — Qiao et al., 2024; <https://github.com/zrqiao/NeuralPLexer>.
  The co-folding + affinity frontier.
- **cuDNN** developer guide (`cudnnConvolutionForward`) — what our hand-rolled conv
  loop maps onto in production.
