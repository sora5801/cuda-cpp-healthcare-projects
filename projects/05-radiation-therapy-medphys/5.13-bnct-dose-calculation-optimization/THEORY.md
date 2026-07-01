# THEORY — 5.13 BNCT Dose Calculation & Optimization

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use. This is a **reduced-scope teaching
> Monte Carlo**: a 1-D, two-energy-group slab model, not a validated transport
> code. The full clinical approach is described in §7._

---

## 1. The science

**Boron Neutron Capture Therapy (BNCT)** is a binary radiotherapy. Step one: give
the patient a boron carrier drug (clinically BPA or BSH) that concentrates
non-radioactive **¹⁰B** preferentially in tumor cells. Step two: irradiate the
tumor region with a beam of low-energy (epithermal) **neutrons**. Neutrons carry
no charge, so they penetrate tissue and slow down ("thermalize") through elastic
collisions, mostly with hydrogen. When a **thermal** neutron (kinetic energy
~0.025 eV) meets a ¹⁰B nucleus, it triggers

$$ {}^{10}\mathrm{B} + n \;\rightarrow\; {}^{7}\mathrm{Li} + {}^{4}\mathrm{He}\,(\alpha) + 2.79~\mathrm{MeV}. $$

The α particle (~1.47 MeV) and the ⁷Li recoil (~0.84 MeV) are **high-LET**
(densely ionizing) and travel only **~5–9 µm** in tissue — about **one cell
diameter**. So if the ¹⁰B sits inside a tumor cell, the lethal energy is
deposited *inside that cell* and spares its neighbors. That cell-scale
selectivity is what makes BNCT attractive for infiltrative tumors (glioblastoma,
head-and-neck recurrence).

The catch for dosimetry: the neutron beam also deposits dose by **other**
reactions everywhere in healthy tissue. The total absorbed dose is the sum of
**four components**, each with a different biological effectiveness:

| Component | Reaction | Character | Weight |
|-----------|----------|-----------|--------|
| **Boron** | ¹⁰B(n,α)⁷Li | high-LET, therapeutic | CBE ≈ 3.8 |
| **Nitrogen** | ¹⁴N(n,p)¹⁴C (0.626 MeV proton) | high-LET background | RBE ≈ 3.2 |
| **Gamma** | ¹H(n,γ)²H (2.22 MeV photon) | low-LET background | RBE ≈ 1.0 |
| **Fast** | ¹H(n,n′)p recoil protons | high-LET background | RBE ≈ 3.2 |

The clinically meaningful quantity is the **CBE/RBE-weighted biological dose**
(units "Gy-Eq"), because 1 Gy of high-LET boron dose kills far more cells than
1 Gy of low-LET gamma. This project computes all four component doses vs. depth
and the weighted biological dose, on both CPU and GPU, and checks they agree
exactly.

## 2. The math

**Transport.** A neutron's fate is governed by the linear Boltzmann transport
equation for the angular flux ψ(**r**, E, **Ω**). In full generality that is a
7-D integro-differential equation (3 space, 1 energy, 2 angle, 1 time). We do
**not** solve it directly; Monte Carlo *samples* particle histories whose
statistics converge to its solution.

**Cross sections.** The probability of an interaction per unit path length is the
**macroscopic cross section** Σ = N·σ (units 1/cm), where N is the number density
(nuclei/cm³) and σ the microscopic cross section (cm², usually quoted in barns,
1 b = 10⁻²⁴ cm²). Each reaction channel has its own Σ:

- fast scatter Σ_s,fast; thermal scatter Σ_s,th;
- thermal capture on boron Σ_a,B, on nitrogen Σ_a,N, on hydrogen Σ_a,H.

The **mean free path** to the next interaction is λ = 1/Σ_tot, and the distance
to that interaction is exponentially distributed. Sampling it by inversion:

$$ s = -\frac{\ln \xi}{\Sigma_\text{tot}}, \qquad \xi \sim U(0,1). $$

**Which reaction happens** is chosen by the cross-section shares. At a thermal
interaction, P(scatter) = Σ_s,th / Σ_tot; on capture, the capturing nuclide is
sampled proportional to its Σ_a share:

$$ P({}^{10}\mathrm{B}) = \frac{\Sigma_{a,B}}{\Sigma_{a,B}+\Sigma_{a,N}+\Sigma_{a,H}}, \; \text{etc.} $$

**Boron's dominance.** σ_a(¹⁰B) ≈ 3837 b at 0.025 eV vs. σ_a(¹⁴N) ≈ 1.83 b and
σ_a(¹H) ≈ 0.332 b — a factor of thousands. Even a few tens of ppm of ¹⁰B make
Σ_a,B comparable to or larger than the tissue background, so most thermal
captures go to boron *where the drug is*. That single number is the whole
physical basis of BNCT.

**Dose.** Absorbed dose is energy deposited per unit mass, D = E/m (1 Gy = 1
J/kg). We tally energy per depth bin in integer keV quanta and scale to Gy by a
single documented constant `gray_per_keV` (a teaching scale, not a real
mass/density calculation). The weighted biological dose is

$$ D_\text{bio} = \sum_c w_c\, D_c, \quad w \in \{\text{CBE}_B, \text{RBE}_N, \text{RBE}_\gamma, \text{RBE}_\text{fast}\}. $$

## 3. The algorithm

We use **two energy groups** (fast, thermal) in a **1-D slab** of thickness L
split into `n_bins`. A monodirectional beam enters at depth 0. One neutron
history (`simulate_neutron` in `bnct_physics.h`):

```
born FAST at z = 0, moving +z
FAST loop:
    s = -ln(xi)/Sig_s_fast ;  z += s
    if z >= L or z < 0:  leak out -> history ends
    if rand < p_thermalize:  become THERMAL (break)
    else:  deposit Q_fast to DC_FAST at bin(z);  keep going fast
THERMAL loop  (Sig_tot = Sig_s_th + Sig_a_B + Sig_a_N + Sig_a_H):
    s = -ln(xi)/Sig_tot
    z += (+/- s)         # 1-D isotropic surrogate: diffuse both ways
    if z out of slab:  leak out -> history ends
    if rand < Sig_s_th/Sig_tot:  scatter -> keep walking
    else:  CAPTURE -> sample nuclide by Sigma_a share, deposit its Q -> end
```

**Complexity.** Let H = `n_histories` and `k` the average number of interaction
steps per history (tens, here). Serial cost is **O(H·k)** transport work plus
**O(H·k)** tally updates; memory is **O(DC_COUNT · n_bins)** for the tally
(tiny). The histories are **embarrassingly parallel** — no history reads another
history's state — so the parallel *work* is the same O(H·k) but the *depth* is
just O(k) (one history's chain), spread across H independent threads. Arithmetic
intensity is low (a handful of `log`, compares, and one atomic per deposit), so
the kernel is latency/branch-bound, not compute-bound — see §4.

## 4. The GPU mapping

This is the canonical **Monte-Carlo pattern** from `docs/PATTERNS.md §1`:
*per-thread RNG + atomic scoring*, exemplified by flagship **5.01** (photon MC).

- **Thread-to-data mapping.** One thread simulates one (or, via a **grid-stride
  loop**, several) neutron histories. History index
  `i = blockIdx.x*blockDim.x + threadIdx.x`, then `i += blockDim.x*gridDim.x`.
- **Launch config.** `block = 256` threads (multiple of the 32-lane warp, good
  occupancy on sm_75–sm_89). `grid = 1024` fixed blocks; the grid-stride loop
  covers any H with one launch — no host-side chunking.
- **Per-thread RNG.** Each thread seeds its own `splitmix64` stream from `(seed,
  i)` (`rng_seed`). Because the RNG lives in the **shared `__host__ __device__`
  header** `bnct_physics.h`, the CPU reference reproduces the *identical*
  histories → verification is **exact**, not statistical.
- **Memory hierarchy.** The tally is small (`DC_COUNT × n_bins` = 80 cells) and
  written by many threads, so it lives in **global memory** and is updated with
  `atomicAdd`. The per-history deposit buffer `dep[]` is per-thread local memory
  (registers/local). No shared memory is needed at this size; a production code
  with a large voxel grid would use shared-memory sub-tallies to cut atomic
  contention.
- **Divergence.** Different neutrons leak / scatter / capture-by-B/N/H after
  different step counts, so warp lanes diverge and finish at different times.
  This is *the* MC-on-GPU challenge. Mitigation (used by real codes, left as an
  exercise here): sort/bin particles by material or energy group into batches so
  a warp processes similar histories together.

```
grid (1024 blocks) x block (256 threads)
   thread i ── history i ── history i+stride ── ...      (grid-stride)
                 │
                 ├─ rng_seed(seed, i)         # private, reproducible stream
                 ├─ simulate_neutron(...)     # shared HD transport (CPU==GPU)
                 └─ for each deposit:
                        atomicAdd(&tally[comp*n_bins + bin], keV)   # global mem
```

## 5. Numerical considerations

- **Precision.** Transport arithmetic (free paths, comparisons) is `double` so the
  CPU (x87/SSE) and GPU agree bit-for-bit on the *branch decisions*. The RNG is
  pure 64-bit integer math — identical on both.
- **Determinism via integer tallies.** The dose is accumulated in **integer keV
  quanta**. Integer addition is associative and commutative, so many threads
  doing `atomicAdd` in a nondeterministic order produce the **same** sum every
  run and the **same** sum as the serial CPU `+=`. A *floating-point* dose tally
  would **not** have this property (float addition is not associative), so its
  GPU result would jitter in the last bits and could not be checked to zero
  tolerance. This is the same lesson as flagship 5.01 (`docs/PATTERNS.md §3`).
- **Race conditions.** The only shared writes are the tally cells, all through
  `atomicAdd` — no data races. Each history's RNG state and `dep[]` are private.
- **Guards.** Loops are capped at 100000 steps to prevent a pathological infinite
  walk; free paths use `xi = 1 - U` ∈ (0,1] so `log` never sees 0.

## 6. How we verify correctness

- **Independent baseline.** `src/reference_cpu.cpp` runs the *same* histories
  serially with plain `+=`. Agreement between an independent serial
  implementation and the parallel-with-atomics GPU implementation is strong
  evidence both are correct.
- **Tolerance = 0 (exact).** Because both sides run identical integer histories,
  we assert the per-component, per-bin tallies match **exactly** (zero
  mismatches). This is the strongest verification tier in `docs/PATTERNS.md §4` —
  available precisely because we chose integer quanta and a shared RNG.
- **Physical sanity checks (science, not just CPU==GPU).** The **boron depth-dose
  curve rises to a sub-surface peak then falls** — the expected "thermal-neutron
  build-up": fast neutrons must slow down before boron can capture them, so the
  thermal flux (and boron dose) peaks a couple cm deep. Boron's share of captures
  scales with Σ_a,B as expected. These recover known BNCT behavior, validating
  the model, not just the port.
- **Edge cases.** The loader rejects non-physical inputs (non-positive geometry,
  `p_thermalize` outside [0,1], all-zero capture cross sections, zero histories).

## 7. Where this sits in the real world

This is a **reduced-scope teaching version**. A clinical/validated BNCT dose
engine differs on essentially every axis:

- **Geometry:** full 3-D voxelized patient anatomy from CT (not a 1-D slab),
  with a per-voxel ¹⁰B concentration map from PET/pharmacokinetics.
- **Energy:** continuous-energy cross sections from **ENDF/B-VIII.0** with
  hundreds of nuclides and resonance treatment (not two flat groups).
- **Angle/kinematics:** correct scattering angular distributions and secondary
  particle transport (α, ⁷Li, recoil protons) — we deposit reaction energy
  locally ("kerma approximation").
- **Codes:** **OpenMC** (open-source, GPU-capable via OpenMP offload),
  **MCNP**, **GATE 10**, **PHITS**; deterministic Sₙ (discrete-ordinates)
  transport is used for fast approximate flux maps. Real GPU implementations
  keep cross-section tables in texture memory and sort particles by material to
  fight warp divergence — exactly the wrinkle noted in §4.
- **Optimization:** treatment planning jointly optimizes beam direction, beam
  spectrum, and boron-carrier dosing/timing against a CBE-weighted objective —
  the "& Optimization" half of the project title, sketched here only via the
  weighted-dose readout.

Our model keeps the two ideas that matter pedagogically: (1) boron's giant
capture cross section produces cell-scale therapeutic selectivity, and (2) the
total dose is a CBE/RBE-weighted sum of physically distinct components.

---

## References

- **OpenMC** — <https://github.com/openmc-dev/openmc> (and its validation tests):
  the reference open-source neutron/photon Monte Carlo; read its transport loop
  and tally machinery to see the production version of §3–§4.
- **GATE 10** — <https://github.com/OpenGATE/opengate>: Geant4-based, includes
  neutron transport used for BNCT studies.
- **COMPASS BNCT MC** — Nature Sci. Rep.,
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC10366114/>: a GPU-accelerated BNCT
  dose engine; good for the clinical dose-component breakdown and CBE handling.
- **OpenMC MeVisLab BNCT pipeline** —
  <https://www.hplpb.com.cn/en/article/doi/10.11884/HPLPB202537.250246>: an
  end-to-end CT→dose BNCT workflow.
- **ENDF/B-VIII.0** — <https://www.nndc.bnl.gov/endf/>: the evaluated
  cross-section library real codes read; the source of the σ values we
  approximated.
