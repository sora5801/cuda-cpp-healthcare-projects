# THEORY — 5.7 Brachytherapy Dose & Source Modeling

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See
> [README.md](README.md) for the quick tour and build steps.
>
> _Educational only — not for clinical use. All data here is synthetic._

---

## 1. The science

**Brachytherapy** ("short-distance" therapy) treats cancer by placing sealed
radioactive **sources** directly inside or next to the tumor — prostate, cervix,
breast, skin. Because the source sits *inside* the target, dose falls off
extremely steeply (`~1/r²`) with distance: the tumor gets a high dose while
nearby healthy tissue is spared. Two delivery styles dominate:

- **HDR (high dose rate):** a single strong source (usually **Ir-192**, ~10 Ci)
  is pushed by a robotic *afterloader* along implanted catheters, **pausing at
  many "dwell positions"** for computer-optimized times. One physical source,
  stepped to 20–100 positions, sculpts the dose.
- **LDR (low dose rate):** many weak permanent **seeds** (**I-125**, **Pd-103**)
  are implanted and left in place.

The clinical question this project answers: **given the source type and where it
dwells, what is the dose everywhere in the patient?** That 3-D dose map is what a
planner checks against tumor-coverage and organ-sparing goals.

The clinical standard for computing it is the **AAPM TG-43** formalism (Task
Group 43). Rather than simulate individual photons, TG-43 gives the dose rate
around a source as a **product of a few pre-measured factors** — fast,
reproducible, and the basis of essentially every commercial brachytherapy
planning system. This project implements TG-43 on the GPU.

## 2. The math

TG-43 places the source at the origin with its long axis along `z`. A field
point **P** is described in polar coordinates `(r, θ)`: `r` is the distance from
the source center, `θ` the angle from the long axis. The **dose rate** at P is

```
             G_L(r, θ)
Ḋ(r,θ) = S_K · Λ · ───────────── · g_L(r) · F(r,θ)
             G_L(r₀, θ₀)
```

with the reference point `r₀ = 1 cm`, `θ₀ = 90°` (the transverse axis). Symbols:

| Symbol | Name | Units | Role |
|---|---|---|---|
| `S_K` | air-kerma strength | `U = µGy·m²/h` | how "strong" the source is |
| `Λ` | dose-rate constant | `cGy·h⁻¹·U⁻¹` | dose rate at `(r₀,θ₀)` per unit `S_K` |
| `G_L(r,θ)` | geometry function | `cm⁻²` | inverse-square, corrected for a source of length `L` |
| `g_L(r)` | radial dose function | — | attenuation + scatter along the transverse axis; `g_L(1cm)=1` |
| `F(r,θ)` | 2-D anisotropy function | — | falloff toward the source poles; `F(r,90°)=1` |

The **line-source geometry function** for an active length `L`:

```
          β / (L · r · sinθ)        θ ≠ 0, 180°   (β = angle subtended by L at P)
G_L(r,θ) =
          1 / (r² − L²/4)           on the long axis (θ = 0, 180°)
```

As `L → 0`, `G_L → 1/r²` — the **point-source inverse-square law**. `g_L` and `F`
are supplied as small **measured tables** (per source model, from TG-43 consensus
data); we interpolate them (1-D linear for `g_L`, bilinear for `F`).

For a **multi-dwell plan**, dose is the **superposition** over dwells `d`:

```
Ḋ_total(P) = Σ_d  weight_d · Λ · [G_L/G_L,ref](r_d, θ_d) · g_L(r_d) · F(r_d, θ_d)
```

where `(r_d, θ_d)` are P's coordinates relative to dwell `d`, and `weight_d`
folds `S_K` and the relative dwell time. **Output:** a 3-D grid of dose rates
(cGy/h); multiply by treatment time for absolute dose (cGy).

## 3. The algorithm

```
for each voxel v in the dose grid:          # N_vox of them
    acc = 0
    for each dwell d in the plan:           # N_dwell of them
        (r, θ) = polar coords of v w.r.t. d
        acc += weight_d · Λ · G_ratio(r,θ) · g_L(r) · F(r,θ)   # tg43_physics.h
    dose[v] = acc
```

- **Complexity:** `O(N_vox · N_dwell)`. A prostate HDR plan is easily
  `256³ ≈ 1.7e7` voxels × `~50` dwells ≈ `8e8` evaluations — seconds on a CPU,
  milliseconds on a GPU.
- **Data-access pattern:** every voxel reads the *same* small source tables and
  the *same* dwell list; it writes exactly one output. This is a **gather from
  shared read-only tables** with **no voxel-to-voxel dependency** — a textbook
  parallel-map (PATTERNS.md §1, "independent jobs + constant-memory tables").
- **Arithmetic intensity** is high: each voxel does dozens of FLOPs (sqrt, atan2,
  interpolation) per dwell but touches only cached constant memory and one global
  write, so it is **compute-bound, not bandwidth-bound** — ideal for the GPU.

## 4. The GPU mapping

**One thread per voxel.** Flatten the `nx·ny·nz` grid to `N` voxels; launch
`ceil(N / 256)` blocks of `256` threads. Thread `t` owns flat voxel index
`i = blockIdx.x·blockDim.x + threadIdx.x`, decoded back to `(ix,iy,iz)` with the
same x-fastest layout the CPU uses (so voxel `i` is identical on both sides).

- **Block size 256:** a multiple of the 32-lane warp; 8 warps/block give the
  scheduler enough in-flight work to hide the latency of the double-precision
  `sqrt`/`atan2`/`acos` in the inner loop. Plenty of blocks stay resident for
  good occupancy on `sm_75…sm_89`.
- **Constant memory for the tables (the key choice).** The `SourceModel` (Λ, `L`,
  `g_L`, `F`) and the dwell list are read by *every* thread but never change
  during the launch, so they live in `__constant__` memory (`kernels.cu`). When a
  warp reads the *same* address — exactly our pattern, since all 32 lanes iterate
  the identical `c_dwells[k]` — the constant cache **broadcasts** it in a single
  transaction. Putting these in plain global memory would replay the same reads
  32× per warp. Budget: `SourceModel` (~7.4 KB) + `Dwell[64]` (~2 KB) ≪ 64 KB.
- **No shared memory, no atomics.** Each voxel is owned by exactly one thread, so
  there is never a write conflict — the output is naturally race-free and
  deterministic (contrast project 5.01's Monte-Carlo tally, which *does* need
  atomics because many photons hit the same bin).
- **Registers:** the per-voxel accumulator and coordinates live in registers; the
  inner loop is short and unrolled by the compiler.

```
        dose grid (nx·ny·nz voxels)                 constant memory (read-only,
   ┌───────────────────────────────────┐             broadcast to every warp)
   │ v0  v1  v2  ...            v(N-1)  │            ┌──────────────────────────┐
   └──┬───┬───┬─────────────────┬───────┘            │  SourceModel: Λ, L,      │
      │   │   │                 │                    │    g_L(r) table,         │
   thread thread thread  ...  thread                 │    F(r,θ) table          │
      │   │   │                 │                    │  Dwell[]: x,y,z,weight   │
      └───┴───┴────► for k in dwells: ───────────────┤  (all threads read same  │
             acc += TG43(source, dwell[k], voxel)    │   entries in lockstep)   │
                     one global write: dose[i]        └──────────────────────────┘
```

**Why the catalog mentions cuRAND / texture memory (and why we do not use them
here).** The catalog's GPU note anticipates the *full* scope:
- **cuRAND** is for the **Monte-Carlo** brachytherapy path (MBDCA), which samples
  photon interactions — that is project 5.01/5.10 territory, not TG-43.
- **Texture memory** is an alternative home for the 2-D `F(r,θ)` table with
  *hardware* bilinear interpolation. We interpolate in software instead so the
  math is *bit-identical* to the CPU reference (hardware texture filtering uses
  reduced-precision fixed-point weights, which would break exact verification).
  Texture filtering is left as an exercise (README §Exercises).
- **Warp-level reduction** would matter if we parallelized *within* a voxel over
  thousands of dwells; with ≤64 dwells a simple per-thread loop is faster and
  clearer.

## 5. Numerical considerations

- **Precision — FP64 in the core, FP32 stored.** `tg43_physics.h` computes in
  `double` (the geometry function's `atan2`/subtraction and the table
  interpolation are sensitive near the source and on-axis). We accumulate the
  per-voxel sum in `double` on **both** CPU and GPU, then store `float` dose. Modern
  GPUs run FP64 slower than FP32, but correctness-you-can-verify beats speed here;
  a production kernel might use FP32 with error analysis (an exercise).
- **The on-source singularity.** `G_L` and `g_L` diverge as `r → 0`. TG-43 is
  simply **not defined inside the source**; we floor `r` to `1e-4 cm` so the
  function stays finite. That is why the demo's "max dose" lands on the exact
  voxel where a dwell sits — an honest artifact, not a bug. Real planning systems
  mask or cap dose within the source volume.
- **The on-axis branch.** At `θ → 0/180°`, `sinθ → 0` and the general line-source
  formula divides by zero; we switch to the closed-form long-axis expression
  `1/(r² − L²/4)` (guarding `r ≤ L/2`). This branch is in the shared header, so
  CPU and GPU take it identically.
- **Determinism.** No atomics, no reordered reductions: each thread computes one
  voxel with a fixed dwell-loop order, the same order the CPU uses. The parallel
  result is therefore **bit-reproducible run to run** and matches the serial one.

## 6. How we verify correctness

- **Independent serial reference.** `src/reference_cpu.cpp` computes the same dose
  with plain nested loops — obviously correct, no parallelism. When the parallel
  GPU agrees with it, we trust the GPU.
- **Shared math ⇒ exact agreement.** Both call the *same* `dose_rate_one_dwell()`
  from `tg43_physics.h` with the *same* `double` accumulation order, so there is
  no algorithmic gap — only fused-multiply-add rounding could differ. In practice
  the observed error is **exactly 0** on this plan.
- **Tolerance and why.** Dose spans orders of magnitude across the grid, so a
  single absolute tolerance is meaningless (a `1e-4` error is huge in a cold voxel,
  trivial in a hot one). We bound the **relative** error per voxel at `1e-5`
  (PATTERNS.md §4). The demo prints `max_rel_err` on stderr.
- **Physical sanity checks (the science, not just CPU==GPU):**
  - `G_L(r,θ) → 1/r²` as `L → 0` — the point-source limit (try `L=0` in the
    sample; the geometry function switches to exact inverse-square).
  - `F(r,90°) = 1` and `g_L(1cm) = 1` by construction — the transverse reference.
  - The center-row profile is **symmetric** and falls off ~`1/r²` — visible in
    `expected_output.txt`.

## 7. Where this sits in the real world

This is a **faithful but reduced-scope** TG-43 implementation. What production
systems add:

- **Real consensus data.** Commercial planners load the AAPM TG-43U1 tables for
  each specific source model. Our `g_L`/`F` are plausible *shapes*, not measured
  values — swap in real tables via `data/sample/plan_sample.txt`.
- **Full 2-D anisotropy + 1-D fallback**, careful near-source handling, and
  active-length orientation **per dwell** (real catheters curve; we assume all
  dwells share a `+z` axis).
- **Beyond TG-43 — heterogeneity.** TG-43 assumes an infinite water phantom. Real
  patients have bone, air, and applicator **shielding**. **Model-Based Dose
  Calculation Algorithms (MBDCA)** — *Acuros BT* (deterministic Boltzmann
  transport, cf. project 5.06) and *Monte Carlo* (EGSnrc **BrachyDose**,
  **TOPAS-BrachyDose**, cf. projects 5.01/5.10) — correct for this. They need the
  particle-transport GPU infrastructure those projects build; TG-43 is the fast
  first pass.
- **Real-time visualization.** Overlaying live dose on a **TRUS** ultrasound feed
  during a prostate implant demands < 100 ms latency — precisely the regime where
  the millisecond GPU TG-43 kernel earns its keep.

---

## References

- **M.J. Rivard et al., "Update of AAPM Task Group No. 43 Report (TG-43U1)",**
  *Med. Phys.* 31(3), 2004 — the formalism and symbol definitions implemented here.
- **AAPM TG-43 consensus source data** — <https://www.aapm.org/pubs/reports/> —
  the real `Λ`, `g_L(r)`, `F(r,θ)` tables per source.
- **PyTG43** (<https://github.com/GregSal/PyTG43>) — a readable Python TG-43
  calculator; good for cross-checking the formulas.
- **matRad** BT module (<https://github.com/e0404/matRad>) — open MATLAB
  brachytherapy dose + inverse planning; shows how dwell-weight optimization sits
  on top of the dose engine (project 5.2).
- **EGSnrc BrachyDose** (<https://github.com/nrc-cnrc/EGSnrc>) and
  **TOPAS-BrachyDose** (<https://github.com/topasmc>) — Monte-Carlo BT for the
  MBDCA/heterogeneity path this project deliberately omits.
