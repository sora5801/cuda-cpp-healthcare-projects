# THEORY — 2.21 Protein-Nucleic Acid Docking & Co-Folding

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

> **Scope.** This is a **reduced-scope teaching version** of the catalog frontier.
> The catalog's headline — co-folding a protein-RNA/DNA complex with an all-atom
> diffusion model — is a multi-GPU deep-learning system. We teach the classical
> kernel underneath it: **rigid-body docking by exhaustive pose scoring**, i.e.
> the catalog's named *"protein-NA interface scoring."* §7 connects the toy to the
> frontier.

---

## 1. The science

Proteins and nucleic acids form **complexes** that run much of biology: a
transcription factor clamped on its DNA operator, a Cas9 protein gripping its
guide RNA and target DNA, an RNA-binding protein recognising a hairpin. To
understand or design these, we need the **3-D structure of the bound complex** —
not just the two partners apart, but *how they meet*: which surfaces touch, which
charges pair, how snugly the shapes interlock.

Computationally there are two halves:

1. **Co-folding / structure prediction** — predict the folded 3-D shape of each
   partner *and* their relative placement, jointly. Modern tools (AlphaFold3,
   Boltz-1, RoseTTAFold2NA) do this with learned diffusion models.
2. **Docking** — given (approximately rigid) 3-D structures of both partners,
   find the relative **pose** (orientation + position) that best fits them
   together, and **score** how good that fit is.

This project implements a clean, exact, GPU-parallel version of **(2)**, because
it is the part you can teach without a black box: it is geometry and a scoring
function, not a trained network. The biological intuition we encode is
**complementarity** — a good interface has matching *shape* (atoms pack closely
without overlapping) and matching *electrostatics* (a protein's positive patch
faces the nucleic acid's negative phosphate backbone). The docking score rewards
both and punishes **clashes** (atoms forced to overlap).

```
        protein surface (fixed)             one candidate pose of the ligand
        + - +                                      o   o   o      <- nucleic-acid
        + 0 -        <-- charged "pocket"          o   o   o         fragment
        + - -                                      o   o   o
        --------- z = 0 plane ---------     slid + rotated over a grid of poses;
                                            each pose gets one interface score
```

## 2. The math

**Inputs.**
- A **protein**: `Np` atoms, each a point `pᵢ = (xᵢ, yᵢ, zᵢ)` with a formal charge
  sign `qᵢ ∈ {−1, 0, +1}`. Fixed in space.
- A **ligand** (nucleic-acid fragment): `Nl` atoms `lⱼ` with charges, given in a
  reference frame.
- A set of **rigid poses**. A pose is a rotation `R ∈ SO(3)` and a translation
  `t ∈ ℝ³`; it moves ligand atom `lⱼ` to `R·lⱼ + t`.

All coordinates are stored as **fixed-point integers** in units of
*milli-Ångström* (`COORD_SCALE = 1000`, so `1500` means `1.5 Å`). This is exact
and central to the numerics (§5).

**Pairwise potential.** For a transformed ligand atom `l'ⱼ = R·lⱼ + t` and protein
atom `pᵢ`, let `d² = ‖pᵢ − l'ⱼ‖²` (an integer, in milli-Å²). The pair score is

```
                ⎧ −clash_pen                       if d² <  clash_r2     (overlap)
  φ(pᵢ, l'ⱼ) =  ⎨ contact_w − elec_w · (qᵢ·qⱼ)     if clash_r2 ≤ d² < contact_r2
                ⎩ 0                                 if d² ≥ contact_r2    (too far)
```

with all five constants positive integers. The middle case is the **interface
shell**: a contact bonus plus an electrostatic term. Because `qᵢ·qⱼ ∈ {−1,0,+1}`,
opposite charges (`−1`) *add* `elec_w` (attractive, favourable), like charges
(`+1`) *subtract* it (repulsive).

**Pose score.** Sum the potential over all pairs:

```
  S(R, t) = Σᵢ Σⱼ φ(pᵢ, R·lⱼ + t)
```

an integer (it never overflows int64 for realistic sizes; see §5). **Objective:**
find the pose maximising `S` — and, for teaching, report the top-K:

```
  (R*, t*) = argmax over enumerated poses of  S(R, t)
```

## 3. The algorithm

We make the pose space **discrete and finite** so we can enumerate it:

- **Orientations.** The **24 proper rotations of a cube** (the rotation subgroup
  of the octahedral group). Their matrices have entries in {−1, 0, +1}, so
  rotating an integer coordinate yields an integer — no trigonometry. We generate
  them by *closure*: start from the identity and the three 90° face rotations,
  multiply until the set stops growing (it stabilises at 24).
- **Translations.** A regular 3-D lattice: `tx = tx0 + ix·step` for `ix ∈ [0,nx)`,
  and likewise y, z. Total poses `P = nx·ny·nz·24`.

A flat index `p ∈ [0, P)` decodes to `(rotation r, ix, iy, iz)` by repeated
divmod (`decode_pose`, used identically by CPU and GPU). The search:

```
  for each pose p in [0, P):
      (R, t)  = decode_pose(p)
      S[p]    = Σⱼ Σᵢ φ(pᵢ, R·lⱼ + t)      # score_pose(): O(Np·Nl)
  return argmax S  and the top-K
```

**Complexity.**
- Serial: `O(P · Np · Nl)` integer pair tests. For the committed sample
  `P=648, Np=25, Nl=9` → ~146k pair tests (instant). Realistically `P` is
  millions and `Np` thousands, so this is the cost that matters.
- Parallel: the `P` pose scores are **independent**. Work is still `O(P·Np·Nl)`;
  the *depth* (critical path) is just one pose score, `O(Np·Nl)`, plus the final
  reduction to find the max. That gap between work and depth is exactly what a GPU
  exploits.
- **Arithmetic intensity.** Each pair does a handful of integer ops on two atoms
  re-read from memory; protein/ligand atoms are reused across all `P` poses, so
  they cache extremely well (high reuse, the GPU-friendly case).

## 4. The GPU mapping

**Thread-to-data map.** One thread scores one pose: thread
`(blockIdx.x, threadIdx.x)` starts at `p = blockIdx.x·blockDim.x + threadIdx.x` and
strides by the total thread count (a **grid-stride loop**) until `p ≥ P`. Each
thread decodes its pose, runs `score_pose` over all `Np·Nl` atom pairs, and writes
one `int64` to `out[p]`. No two threads write the same slot → **no atomics, no
shared-memory reduction in the kernel**, fully independent and deterministic.

**Launch configuration.** `block = 256` threads (a multiple of the 32-lane warp;
8 warps give the scheduler enough latency-hiding work; good occupancy on
sm_75…sm_89). `grid = min(⌈P/256⌉, 1024)` blocks — capped, with the grid-stride
loop covering any larger `P`. The same block size as every other flagship.

**Memory hierarchy.**
- **Constant memory** holds the 24 rotation matrices (`c_rots`). Every thread reads
  the matrix for *its* orientation; the constant cache **broadcasts** one address
  to a whole warp in a single transaction — ideal for small, read-only, shared
  data. (Same trick as the query in `1.12`.) Footprint: 24·9·4 B ≈ 864 B, far
  inside the 64 KB bank.
- **Global memory** holds the protein and ligand atom arrays (read-only, marked
  `__restrict__`). They are reused across poses, so the L2 cache absorbs most
  traffic.
- **Registers** hold the running `int64` accumulator and the transformed ligand
  atom — the hot state stays on-chip.

```
         grid (1-D)                 each thread:
   ┌───────┬───────┬─── …            pose p ──► decode_pose ──► (R, tx,ty,tz)
   │block 0│block 1│                 R from constant cache (broadcast)
   └───┬───┴───────┘                 loop Nl ligand atoms  (apply_pose, int)
       │ 256 threads                   loop Np protein atoms (pair_score, int)
       ▼                              out[p] = Σ  (one int64, no contention)
   thread t  ── scores pose (block·256 + t),  then strides by gridDim·256
```

**No CUDA library is used.** Unlike `8.03` (cuFFT) or `2.06` (cuSOLVER), every
operation here is hand-written integer arithmetic, so there is no library to
explain — and that is deliberate: it keeps the math identical to the CPU (§5).
The natural *next* step, an FFT over translations (ZDOCK-style), **would** use
`cuFFT`; §7 and Exercise 2 sketch it.

## 5. Numerical considerations

**Everything is integer.** Coordinates are int32 fixed-point; squared distances
and score sums are accumulated in **int64**. This is the whole numerical story,
and it buys two things:

1. **No floating-point non-associativity.** A float sum's value depends on the
   order it is added (rounding), so a parallel reduction can disagree with a serial
   one in the last bits. Integer addition is exactly associative and commutative,
   so the pose score is **independent of evaluation order** — the GPU and CPU get
   the *same* integer, and re-running is byte-identical (PATTERNS.md §3).
2. **No FMA divergence.** The GPU fuses multiply-add and may round differently from
   the host compiler; over many ops this drifts (see `10.02`, `14.02`). Integer
   ops have no rounding, so there is nothing to diverge. We choose cube-group
   rotations precisely so rotation stays integer too.

**Overflow safety.** A coordinate fits in ±2.1×10⁹ units (±2.1×10⁶ Å). A per-axis
difference squared is ≤ ~1.8×10¹³; times 3 axes ≈ 5×10¹³, far under int64's
9.2×10¹⁸. Score sums over `Np·Nl` pairs with per-pair magnitude ≤ `clash_pen`
stay tiny by comparison. No overflow for any realistic molecule.

**Race conditions.** None: each thread owns one output slot. There is no shared
accumulator, so no atomics and no reduction race.

**Precision trade-off.** The cost of exactness is **resolution**: fixed-point and
24 orientations are coarse. A finer search (sub-Ångström grid, thousands of
quaternion orientations) needs floating point and re-introduces a *tolerance*
(Exercise 1) — the honest engineering trade we point at, not hide.

## 6. How we verify correctness

The CPU reference `dock_cpu` (`src/reference_cpu.cpp`) is a single readable triple
loop with **no parallelism**. It calls the **same** `score_pose` /
`decode_pose` the GPU calls — those live in `docking_core.h` / `reference_cpu.h`
as `__host__ __device__` inline functions (the HD-macro idiom, PATTERNS.md §2), so
host and device compile *the identical source* for the physics.

`main.cu` therefore checks **exact integer equality** over *every* pose:

- **Tolerance = 0.** We count mismatching pose scores; it must be `0/P`. (Contrast
  the iterative double-precision projects `10.02`/`14.02`, which need a small
  physical tolerance — here the integers make exactness achievable and we use it.)
- **A scientific check, not just CPU==GPU.** The committed sample has a **planted
  native pose** — a charge "lock" on the protein and a complementary "key" ligand,
  positioned so identity-rotation at `t = (0,0,+spacing)` maximises contacts and
  electrostatics with no clashes. The search must return *that* pose as #1. The
  expected output confirms it (`pose 312, rot 0, score 340`, a clear margin over
  the runner-up 240). The charge pattern is deliberately **chiral** so the maximum
  is *unique* — a checkerboard would tie under several cube rotations (Exercise 5).

Why convincing: an independent serial implementation and a parallel GPU
implementation agreeing *exactly* on all 648 poses, *and* both recovering an
answer we planted by hand, is strong evidence the kernel is correct.

## 7. Where this sits in the real world

| Aspect | This teaching project | Production (ZDOCK/PIPER · AlphaFold3/Boltz-1) |
|---|---|---|
| Problem | rigid docking + scoring | full **co-folding** (predict 3-D + bind) |
| Orientations | 24 cube rotations | thousands of quaternions |
| Translations | small integer lattice, brute force | **FFT** over the grid: `O(N³)`→`O(N³ log N)` |
| Scoring | 3-shell integer potential | shape + electrostatics + desolvation; or learned |
| Flexibility | both bodies rigid | side-chain / backbone flexibility; full folding |
| Engine | hand-written integer CUDA | cuFFT + clustering; or diffusion transformers |

The single biggest real-world idea we omit is the **FFT docking trick**: for a
*fixed* orientation, the score as a function of translation is a **correlation** of
the protein and ligand grids, which the convolution theorem evaluates for *all*
translations at once via one forward + one inverse FFT — turning the `O(grid³)`
translational scan into `O(grid³ log grid)`. ZDOCK and PIPER/ClusPro are built on
this; on the GPU it is a `cuFFT` job (see project `8.03` for the API, and
Exercise 2). The learned co-folders go further still: instead of scoring a fixed
shape, they *generate* the bound 3-D structure with an all-atom diffusion model
over a token vocabulary of amino acids **and** nucleotides — that is the catalog's
frontier, and the thing our pose search is a small, transparent ancestor of.

---

## References

- **Chen, Li & Weng, *ZDOCK* (Proteins, 2003)** — FFT rigid-body docking; the
  algorithm our brute-force scan should grow into.
- **Kozakov et al., *PIPER / ClusPro*** — FFT docking with a pairwise potential and
  clustering; closest classical peer to this project's scoring idea.
- **Baek et al., *RoseTTAFold2NA* (Nat. Methods, 2024)** — deep prediction of
  protein-nucleic-acid complexes; how NA tokens and templates enter a network.
- **Abramson et al., *AlphaFold3* (Nature, 2024)** and **Wohlwend et al.,
  *Boltz-1* (2024)** — all-atom diffusion co-folding of protein + RNA/DNA + ligand.
- **Lorenz et al., *ViennaRNA Package 2.0*** — RNA secondary structure (the CPU
  preprocessing step real pipelines run before/around docking).
- **NVIDIA CUDA C Programming Guide** — constant memory & broadcast cache,
  grid-stride loops, the occupancy model behind the §4 launch choices.
