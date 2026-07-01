# THEORY — 5.3 Proton & Heavy-Ion Therapy Dose

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

### Why protons at all?

Radiotherapy kills tumour cells by depositing ionising radiation, but every
healthy cell the beam passes through is also irradiated. **Photon** (X-ray) beams
deposit dose that peaks a couple of centimetres below the skin and then falls off
*slowly* — so tissue *behind* the tumour still gets a substantial dose (an "exit
dose"). **Proton** and **heavy-ion** (e.g. carbon) beams behave completely
differently, and that difference is the whole reason the modality exists.

A charged particle slowing down in matter loses energy per unit path length at a
rate given by the **stopping power** `−dE/dx`. The Bethe formula makes `−dE/dx`
roughly proportional to `1/v²` (inverse square of the particle's velocity). As a
proton slows, it deposits energy *faster and faster*, dumping the bulk of it in a
narrow region right before it stops. Plotted against depth, the dose rises to a
sharp maximum — the **Bragg peak** — and then falls essentially to **zero**,
because past that depth there are no protons left to deposit anything.

```
 dose
  ^
  |                                   ##   <- Bragg peak (protons stop here)
  |                                 ####
  |        entrance plateau       ######
  |   ####################-------########
  |   ####################       ########
  |   ####################       ########|
  +---------------------------------------+---> depth
  0                                 R      (beyond R: ZERO dose)
```

Contrast with a photon beam, whose dose keeps going past the target. The proton's
zero exit dose is what lets a plan **spare organs behind the tumour** — the single
most important fact in the field.

### The knobs a planner turns

- **Range `R`** — the depth of the Bragg peak — is set by the beam **energy**. Higher
  energy → deeper peak. This is how the planner steers dose to the right depth.
- **Lateral position `(x0, y0)`** — modern **pencil-beam scanning (PBS)** machines
  magnetically steer a thin beam across the field, painting the tumour spot by spot.
- **Weight `w`** — how many protons (monitor units, MU) each spot delivers.

A **plan** is a list of thousands of such **spots**. The dose engine's job: given
the plan, compute the 3-D dose everywhere. A single Bragg peak is too narrow to
cover a tumour, so planners stack peaks of several energies — heaviest at the
deepest — into a flat-topped **spread-out Bragg peak (SOBP)**. (Try this with
`make_synthetic.py --ranges`; see README exercise 1.)

### Heavy ions and what we omit

Carbon ions are even sharper laterally and carry higher **linear energy transfer
(LET)**, which raises their **relative biological effectiveness (RBE)** — the same
physical dose does more biological damage. They also undergo **nuclear
fragmentation** (the carbon nucleus shatters, and the fragments deposit a "tail"
of dose *beyond* the Bragg peak). LET, RBE, and fragmentation require full Monte
Carlo and are **out of scope** for this analytic teaching model — §8 says how the
real tools handle them.

---

## 2. The math

We model the dose from one spot as a **separable pencil-beam kernel**: a depth term
times a lateral term. This factorisation *is* the pencil-beam algorithm (PBA).

For a spot with axis `(x0, y0)`, range `R`, weight `w`, the dose at a point
`(x, y, z)` (with the patient surface at `z = z_entry`, so depth `d = z − z_entry`)
is

```
dose_spot(x,y,z) = w · IDD(d; R) · L(x−x0, y−y0; σ(d))
```

**Depth term — integral depth dose `IDD(d; R)`.** Written through the *residual
range* `u = R − d` (how far the protons still have to go):

```
             ⎧ 0                                        if u ≤ 0   (past the range)
IDD(d; R) =  ⎨
             ⎩ (u + w_p)^(1/p − 1)  +  c / (u + w_p)    if u > 0
```

- The **peak term** `(u + w_p)^(1/p − 1)` uses the Bortfeld exponent `p ≈ 1.77`.
  Since `1/p − 1 ≈ −0.435 < 0`, it **blows up (finitely) as `u → 0`** — the sharp
  Bragg peak — and is **small for large `u`** (near entrance). `w_p` (`peak_width`)
  regularises the true integrable singularity at `u = 0` so the peak is finite and
  host/device evaluate it identically.
- The **plateau term** `c / (u + w_p)` (with `c = 0.20`) adds the small, slowly
  varying entrance dose deposited all along the track.
- For `u ≤ 0` the dose is exactly **0** — the hard distal fall-off.

**Lateral term — 2-D Gaussian `L`:**

```
L(Δx, Δy; σ) = 1/(2π σ²) · exp( −(Δx² + Δy²) / (2σ²) ),   σ(d) = σ0 + σ_grow · d
```

The width `σ` **grows with depth** because of multiple Coulomb scattering. The
`1/(2πσ²)` prefactor conserves the in-plane integral (∫∫ L dx dy = 1), so widening
the beam *lowers* the central value rather than adding energy — a real property that
matters for §6.

**The full field** is the superposition (a convolution of the spot map with the
pencil-beam kernel):

```
Dose(x,y,z) = Σ_{s ∈ spots} dose_spot_s(x,y,z)
```

Notation: lengths in cm, weights in arbitrary MU-like units, dose in arbitrary
units (normalised to the peak for display).

---

## 3. The algorithm

```
load Plan {grid, beam, spots[], z_entry}
allocate Dose[nx*ny*nz] = 0
for each voxel (i,j,k):                 # ~10^7 voxels
    (vx,vy,vz) = centre of voxel (i,j,k)
    acc = 0
    for each spot s:                    # ~10^4 spots
        acc += dose_from_spot(beam, s, vx, vy, vz, z_entry)
    Dose[idx(i,j,k)] = acc
report: integral depth-dose IDD(k) = Σ_{i,j} Dose[i,j,k]; Bragg-peak depth = argmax_k IDD
```

`dose_from_spot` is the single per-`(voxel, spot)` evaluation of §2, and it is the
**one function** shared by the CPU reference and the GPU kernel
(`src/proton_physics.h`).

**Complexity.** With `V` voxels and `S` spots the work is `O(V · S)`.

- **Serial (CPU):** one core walks all `V·S` pairs → time `∝ V·S`. For a clinical
  plan (`V ≈ 10⁷`, `S ≈ 10⁴`) that is `10¹¹` evaluations — minutes.
- **Parallel (GPU):** `V` threads each do `O(S)` work → **span** `O(S)` with `O(V)`
  processors (ideally). The embarrassingly parallel voxel loop is the win; the
  per-voxel spot loop stays serial inside the thread.

Both the "for each voxel" (parallel) and "for each spot" (serial) loops appear
literally in the code — `dose_cpu` in `reference_cpu.cpp` and `dose_kernel` in
`kernels.cu`.

---

## 4. The GPU mapping

**Thread ↔ data.** One thread owns one voxel. We flatten the 3-D grid to a 1-D
index with **x fastest**:

```
idx = (k · ny + j) · nx + i
```

so threads with adjacent `i` (adjacent `idx`) write adjacent addresses → **coalesced**
stores. A **grid-stride loop** lets a fixed grid (`blocks × 256` threads) cover any
voxel count:

```
for (idx = blockIdx.x*blockDim.x + threadIdx.x; idx < n_voxels; idx += blockDim.x*gridDim.x)
```

**Memory hierarchy — deliberate choices.**

| Data | Space | Why |
|---|---|---|
| spot list `c_spots[]` | **constant** | read by *every* thread, never written; constant memory's broadcast cache serves one address to a whole warp in a single transaction (like the query fingerprint in flagship 1.12) |
| accumulator `acc` | **register** | private per thread; no sharing, so no atomics and no shared-memory traffic |
| dose volume `d_dose[]` | **global** | one coalesced write per voxel |
| `Grid`, `BeamModel` | passed **by value** | tiny structs land in constant/parameter memory, readable by all threads |

**Why no atomics (contrast with 5.01).** This is the key teaching contrast in GPU
dose calculation:

- The **Monte-Carlo** engine (5.01) **scatters**: random particle histories deposit
  into shared depth bins, so many threads hit the *same* memory → you *need*
  `atomicAdd`, and you accumulate in *integers* to stay deterministic.
- This **analytic** engine **gathers**: each voxel *pulls* from all spots into a
  private register and writes its *own* cell → **no collisions, no atomics**, and
  the result is deterministic for free.

Gather vs. scatter, and register-accumulate vs. atomic-accumulate, are the two
faces of dose on the GPU. (See docs/PATTERNS.md §1.)

**Occupancy / bandwidth.** 256 threads/block = 8 warps; the kernel is
**compute-bound** (lots of `expf`/`powf`), not bandwidth-bound — each voxel reads the
whole constant-memory spot list (cheap, cached) and writes one float. Bigger plans
scale linearly in `V·S`; the GPU's edge over the CPU grows with that product (§7).

---

## 5. Where the constant-memory cap comes from

Constant memory is **64 KB** total. Each `Spot` is 4 floats = 16 B, so 4096 spots
= 64 KB is the hard ceiling; we cap at **2048** (32 KB) for headroom. A real plan
has ~10⁴ spots — beyond the cap — so production engines **tile** the spot list from
global memory into shared memory (load a tile, let the block's threads reuse it,
advance). That is left as README exercise 5; the teaching version keeps the simpler
constant-memory path so the data flow is obvious.

---

## 6. Numerical considerations

- **Precision (FP32).** Clinical GPU dose engines run in single precision for
  speed/memory, and FP32 is ample for a teaching model. We keep *both* sides in FP32
  so the comparison is apples-to-apples (see §7).
- **The lateral-normalisation trap.** Because `σ(d)` grows with depth and `L`
  carries the `1/(2πσ²)` factor, the **on-axis** value *falls* with depth even where
  the slice's total energy is *rising*. Reading the central axis would therefore
  *flatten* the Bragg peak. We instead report the **integral depth dose** — the dose
  summed over the whole lateral plane per slice — where the Gaussian integrates to 1
  and only the true depth shape (the clean Bragg peak) survives. This is exactly the
  IDD a physicist measures with a large parallel-plate chamber in a water tank. The
  plane sum uses a `double` accumulator (a reporting quantity, so extra precision only
  sharpens the plot; it is not part of the CPU==GPU comparison).
- **Determinism.** No atomics, and each voxel sums spots in the **same index order**
  on CPU and GPU, so FP32 rounding is identical → stdout is byte-for-byte
  reproducible (docs/PATTERNS.md §3). Timings go to stderr (they vary; not diffed).
- **Guards.** `σ` is clamped `> 0` (no divide-by-zero); `u ≤ 0` returns exactly 0
  (no `pow` of a negative base); voxels upstream of the surface (`depth < 0`) return 0.

---

## 7. How we verify correctness

Two checks, from cheap to meaningful:

1. **CPU == GPU, voxel-by-voxel.** `max_abs_err = max_v |Dose_GPU[v] − Dose_CPU[v]|`
   must be `≤ 1.0e-4` (absolute dose units). In practice it is `~1.8e-7` — a few FP32
   ULPs. It is **not exactly zero** because `expf`/`powf` have *different last-bit
   implementations* on the host compiler vs. the device (docs/PATTERNS.md §4). We
   verify to a physically negligible tolerance and say so, rather than pretending
   bit-identity.
2. **Bragg-peak depth agrees.** `argmax_k IDD(k)` must be the *same bin* for CPU and
   GPU, and for a single spot it should land **half a voxel proximal to the input
   range** (peak at 11.75 cm for a 12 cm spot on a 0.5 cm grid) — a *science-level*
   check that the depth shape, not just the arithmetic, is right.

This is the standard recipe: a readable serial reference plus a known-answer sanity
check (the recovered range) — the same shape as the analytic checks in 6.04
(Poiseuille) and 13.02 (`AUC ≈ dose/CL`).

**Edge cases exercised:** the ragged last block (grid-stride guard), a voxel behind
the range (hard-zero branch), and the boundary `u → 0` (regularised peak).

---

## 8. Where this sits in the real world

What production tools do that this teaching model does **not**:

| Aspect | This project | Production (matRad, FRED, TOPAS, MOQUI) |
|---|---|---|
| Depth term | regularised Bortfeld surrogate | exact Bortfeld (parabolic-cylinder fn) fit to measured IDDs |
| Lateral term | single Gaussian | **double**-Gaussian (adds the nuclear "halo") or full MC |
| Medium | homogeneous, dose ∝ geometric depth | **CT stopping-power map**; range = *water-equivalent* depth via ray-tracing (texture memory) |
| Physics | analytic only | nuclear **fragmentation**, secondary particles, **LET/RBE** weighting |
| Spots | ≤ 2048 in constant memory | ~10⁴, streamed/tiled from global memory |
| Optimization | fixed weights | **multi-field** + **robust** optimization over 3 mm / 3.5 % range-uncertainty scenarios |
| Convolution | direct real-space sum | often **cuFFT** in k-space for the lateral convolution |
| Ions | protons only | carbon/helium with fragmentation tails (MC-dominated) |

The two production *families* map to two GPU patterns: **analytic PBA** (this
project — gather/convolution, fast, approximate) vs. **Monte Carlo** (project 5.01 —
scatter/atomics, slow, gold-standard). Clinics use PBA for planning speed and MC for
verification. Understanding *both* — and why one uses atomics and the other does not
— is the point of this pair.

> **Not for clinical use.** Everything here is a reduced educational model in
> arbitrary units. It must never inform a real treatment (CLAUDE.md §1, §8).
