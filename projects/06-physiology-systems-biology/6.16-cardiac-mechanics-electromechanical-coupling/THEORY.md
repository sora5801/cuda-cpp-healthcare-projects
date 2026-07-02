# THEORY — 6.16 Cardiac Mechanics & Electromechanical Coupling

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use. All numbers are synthetic._

## 0. What we build, and the honest scope

The catalog entry describes a **3-D nonlinear finite-element** cardiac
electromechanics solver: a stiff ionic + cross-bridge ODE at every Gauss point of
a hyperelastic myocardium mesh, coupled to a global Newton–Raphson equilibrium
solve, closed by Windkessel boundary conditions. That is a multi-week code and it
needs a GPU cluster to be interesting.

This project ships a **reduced-scope teaching version** (CLAUDE.md §13) that keeps
the two things worth learning here:

1. **The coupling chain** — *voltage → calcium → cross-bridges → chamber stiffness
   → pressure → ejection → PV loop* — implemented end to end, just at **0-D** (a
   single representative cell + a lumped ventricle + a Windkessel).
2. **The GPU pattern** the catalog names first: a **batch of independent ODE
   solves, one integration point per thread**. Our integration points are whole
   virtual hearts (a contractility × afterload sweep) instead of Gauss points, so
   the demo runs on any laptop GPU.

§7 describes what the full FEM solver adds.

## 1. The science

A heartbeat is an **electrical** event that becomes a **mechanical** one:

- An action potential depolarises the myocyte membrane.
- Depolarisation opens L-type Ca²⁺ channels; Ca²⁺ floods in and triggers further
  release from the sarcoplasmic reticulum — the **calcium transient** (free
  cytosolic Ca²⁺ spikes ~10× then is pumped back by SERCA).
- Ca²⁺ binds **troponin C**, which moves tropomyosin off the actin binding sites,
  letting **myosin cross-bridges** attach and pull — this is **active tension**.
  The binding is *cooperative* (a few Ca²⁺ ions strongly bias the switch), which is
  why a **Hill curve** describes the steady-state activation.
- The contracting muscle stiffens the ventricular wall. When the chamber pressure
  exceeds aortic pressure, the **aortic valve** opens and blood is ejected; when it
  falls below venous pressure the **mitral valve** opens and the chamber refills.
- Plotting chamber **pressure vs. volume** over one beat traces the **PV loop**;
  its width is the **stroke volume (SV)**, and **ejection fraction EF = SV/EDV** is
  the single most-used clinical index of pump function (normal ~55–70%; a failing
  heart drops below ~40%).

The real-world questions this kind of model answers: *How does reduced
contractility (a weak muscle) lower EF? How does raised afterload (stiff
arteries / hypertension) change the loop and the wall stress? Which therapy moves
the loop back toward normal?*

## 2. The math

**State vector** (per heart) integrated in time `t` [ms]:

```
y = ( Ca , xb , V , Pa )
```

| Symbol | Meaning | Units |
|---|---|---|
| `Ca` | intracellular free calcium | µM |
| `xb` | cross-bridge / active-state fraction | — (0…~1) |
| `V`  | ventricular blood volume | mL |
| `Pa` | arterial (Windkessel) pressure | mmHg |

**Driven calcium transient** (phenomenological difference-of-exponentials, peaking
at `Ca_amp` above rest at phase `t`):

```
Ca*(t) = Ca_rest + Ca_amp · [ e^(−s/τ_decay) − e^(−s/τ_rise) ] / peak ,   s = t − t_activate ≥ 0
```

normalised by its analytic peak so the pulse height is exactly `Ca_amp`.

**Hill activation** (cooperative troponin binding), the steady state of `xb`:

```
xb_ss(Ca) = Ca^n / (Ca50^n + Ca^n)          n = Hill coefficient
```

**Time-varying elastance ventricle** (Suga–Sagawa). The instantaneous chamber
elastance and pressure:

```
E(t)  = Emin + Tref · xb                     [mmHg/mL]
P_lv  = max( E(t) · (V − V0) , 0 )           [mmHg]
```

`Tref` is **contractility**: how stiff the chamber gets at full activation.

**Valves** (linear resistances while open):

```
q_ao = (P_lv − Pa) / R_ao    if P_lv > Pa   else 0     (ejection, mL/ms)
q_mv = (P_ven − P_lv) / R_mv if P_ven > P_lv else 0     (filling,  mL/ms)
```

**Governing ODE** (the coupled system):

```
dCa/dt = (Ca*(t) − Ca) / τ ,   τ = τ_rise while rising, τ_decay while falling
dxb/dt = k_xb · ( xb_ss(Ca) − xb )
dV/dt  = q_mv − q_ao
dPa/dt = ( q_ao − (Pa − P_art_dias)/R_sys ) / C_art     (2-element Windkessel)
```

`R_sys` is **afterload**. Inputs: the `HeartParams` for each heart. Outputs per
heart: EDV, ESV, SV, EF, peak pressure, and a Laplace wall-stress proxy
`σ ∝ P·V^(1/3)`, taken over the last (steady-state) beat.

## 3. The algorithm

For each heart:

1. Initialise `y` at a relaxed, partly-filled state.
2. For `n_beats` cardiac cycles, take `steps_per_beat` **RK4** steps of the coupled
   ODE (four derivative evaluations per step, combined with weights (1,2,2,1)/6,
   local error `O(dt⁴)`).
3. Discard the warm-up beats (the ventricle–Windkessel system relaxes onto a
   **limit cycle** after a few beats), and record min/max `V`, peak `P`, and peak
   stress over the **last** beat.
4. Reduce those to the scalar `CycleResult`.

**Complexity.** One heart costs `Θ(n_beats · steps_per_beat)` RK4 steps, each `O(1)`
work → for the sample, `10 · 8000 = 80,000` steps. The ensemble of `M = nT·nR`
hearts costs `Θ(M · n_beats · steps_per_beat)`.

- **Serial (CPU):** the hearts run one after another → total `Θ(M · steps)`.
- **Parallel (GPU):** the hearts are **independent**, so with `M` threads the
  *depth* is just `Θ(steps)` (one heart's time loop) and the *work* is unchanged.
  This is the classic ensemble/UQ speed-up: throughput scales with `M`.

**Arithmetic intensity / access pattern.** Each thread keeps its entire state in
registers and does pure FLOPs (exp, pow, adds) with **no global-memory traffic**
during the time loop and **no inter-thread communication**. It is compute-bound per
thread; the only memory op is writing one `CycleResult` at the end.

## 4. The GPU mapping

**Thread-to-data mapping.** One thread ↔ one heart:

```
idx = blockIdx.x * blockDim.x + threadIdx.x;   // heart index
if (idx >= M) return;                           // guard the ragged last block
out[idx] = integrate_member(c, idx);            // full multi-beat RK4 in registers
```

**Launch configuration.** `blockDim = 128` threads (a good sm_75–sm_89 occupancy
default), `gridDim = ceil(M / 128)`. Each thread is register/local-memory heavy (it
holds `Ca, xb, V, Pa` plus four RK4 stage structs), so a modest block keeps register
pressure and spills reasonable; 128 balances occupancy against that.

**Memory hierarchy.**
- **Registers / local memory:** the whole ODE state and RK4 temporaries — this is
  where all the work happens, at full register speed.
- **Constant/param memory:** the small `EnsembleConfig` is passed **by value** as a
  kernel argument, so every thread reads the baseline physiology + sweep ranges
  from the fast constant/param path; each thread derives its own `(Tref, R_sys)`
  on the fly via `member_params()` — no per-heart input array to stage.
- **Global memory:** touched exactly once per thread, to write the result.
- There is **no shared memory** and **no bandwidth bottleneck**: this kernel is
  latency/occupancy-bound, not bandwidth-bound.

**No CUDA library needed** for this reduced version — the ODE is hand-rolled RK4 so
the learner sees every operation (no black box). The *full* solver would use
cuSOLVER (dense/sparse Newton linear solves) and cuSPARSE (stiffness SpMV); §7.

```
grid (ceil(M/128) blocks)
┌───────────── block 0 ─────────────┐ ┌──── block 1 ────┐ ...
│ t0  t1  t2  ...  t127              │ │ t128 ...        │
│  │   │   │        │                │ │                 │
│ h0  h1  h2  ...  h127   (each thread = one heart)      │
│  └── full 80,000-step RK4 loop in registers ──┐        │
│                                    write out[h] ▼      │
└───────────────────────────────────────────────────────┘
```

## 5. Numerical considerations

- **Precision: FP64 throughout.** The valve switches and the multi-beat limit-cycle
  settling need double precision; FP32 would visibly bias EF. `sm_75` FP64 is slow
  but this is teaching code, not a benchmark.
- **Stiffness & stability.** A naive "orifice" ventricle (flow `∝ (P−Pa)` with a
  *tiny* fixed resistance) is stiff and blows up under explicit RK4 at `dt = 0.5`
  ms — an early version of this project did exactly that. The **time-varying
  elastance + finite valve resistances** formulation bounds all flows, so explicit
  RK4 is stable at `dt = 0.1` ms. (The lesson: choose a formulation whose fastest
  timescale your explicit integrator can resolve, or go implicit.)
- **No atomics, no reordering.** Each thread writes its own output; there is no
  parallel reduction, so there is no floating-point-summation nondeterminism. The
  program's **stdout is byte-identical every run** (verified); timing goes to stderr.
- **CPU–GPU divergence is real and bounded (this is the key teaching point).** CPU
  and GPU run the *same* RK4 on the *same* `cardiac.h` source, but the GPU
  **contracts** `a*b + c` into a single fused-multiply-add (one rounding) where the
  host compiler emits a multiply then an add (two roundings). Per step the two
  trajectories drift by ~`1e-15`; compounded over 80,000 steps **and** pushed
  through the non-smooth valve/`τ` switches, the recorded scalars differ by
  ~`1e-3`–`1e-2`. The divergence is **largest for peak pressure**, because it is a
  `max` over discrete samples: a sub-microvolt trajectory shift can move *which*
  timestep is the maximum, producing a ~`1e-2` mmHg jump. Measured on this machine:
  worst EF diff ≈ `5e-4`, SV ≈ `7e-4`, peak-P ≈ `3.7e-2`.

## 6. How we verify correctness

- **Independent CPU reference.** `src/reference_cpu.cpp` integrates every heart with
  a plain serial loop. It is compiled by the host compiler and shares only the
  `__host__ __device__` physics header — so agreement is *evidence*, not tautology:
  two different toolchains (nvcc-for-device vs. cl.exe) producing the same PV-loop
  scalars is convincing.
- **Tolerance (documented, physical).** We require the worst per-heart difference in
  EF, SV, and peak pressure to be `≤ 0.1` (percentage-point / mL / mmHg). This is a
  **physical** tolerance chosen to match the FMA/limit-cycle divergence of §5
  (PATTERNS.md §4 category: long iterative solver): relative ~`1e-3` on ~80-mmHg,
  ~100-mL, ~50-% quantities — physiologically negligible. We do **not** claim
  bit-identical results, because they are not.
- **Physical sanity as a second check.** Beyond CPU==GPU, the results must make
  physiological sense, and they do: EDV ~135 mL, EF spanning ~36% (weak) → ~65%
  (strong), monotone in contractility, with peak pressure rising with afterload.
  A number that agreed CPU-to-GPU but gave EF = 200% would still be wrong — the
  physiology check guards against that.
- **Loader edge cases.** `load_ensemble()` throws on missing blocks, non-positive
  `dt`/steps/sweep sizes, `τ_rise ≥ τ_decay`, or non-positive resistances, so a
  typo fails loudly instead of producing silent nonsense.

## 7. Where this sits in the real world

Production cardiac-electromechanics solvers (the catalog's prior art) differ from
this teaching version in scope, not in spirit:

- **Geometry & FEM.** **FEBio**, **simcardems** (FEniCS), **OpenCMISS**, and
  **Chaste** discretise a real 3-D ventricular mesh with hyperelastic finite
  elements. The passive myocardium uses the **Holzapfel–Ogden** (fibre/sheet
  anisotropic) or **Guccione** exponential strain-energy law instead of our scalar
  elastance; **incompressibility** is enforced with a penalty or mixed
  (displacement–pressure) formulation.
- **Active stress/strain.** Active tension is added either to the stress
  (active-stress) or by a multiplicative deformation split (active-strain), driven
  by a full **Rice–Wang–Bers** cross-bridge model rather than our single `xb`.
- **Electrophysiology.** A **monodomain** (or bidomain) reaction–diffusion PDE
  propagates the activation across the tissue, setting *when* each Gauss point
  fires — we simply fix `t_activate`.
- **The solve.** Each timestep is a nonlinear **Newton–Raphson** equilibrium: batch
  the stiff cell ODE at every Gauss point (the GPU-batched CVODE the catalog names),
  assemble the tangent stiffness (cuSPARSE **SpMV**), and solve the linear system
  (cuSOLVER). The natural GPU decomposition is a **two-level grid**: elements outer,
  Gauss points inner, with per-element stiffness accumulated in **shared memory**.
- **Boundary conditions.** The same **Windkessel** we use here closes the real model
  too — that part is faithful.

Our 0-D model is exactly what these tools reduce to at a single material point with
a lumped chamber, which is why it is the right first rung on the ladder.

---

## References

- **Suga H., Sagawa K.** (1974) — instantaneous pressure–volume relationship and
  time-varying elastance; the basis of our ventricle model.
- **Rice J.J., Wang F., Bers D.M., de Tombe P.P.** (2008) — approximate model of
  cooperative cross-bridge dynamics; the full version of our `xb` equation.
- **Holzapfel G.A., Ogden R.W.** (2009) — anisotropic hyperelastic constitutive law
  for passive myocardium (the FEM passive stress).
- **Guccione J.M., McCulloch A.D.** (1991) — exponential passive strain-energy law.
- **Westerhof N. et al.** — the arterial Windkessel and its 2/3/4-element variants.
- **FEBio** <https://github.com/febiosoftware/FEBio> — study tangent-stiffness
  assembly and the nonlinear solve loop.
- **simcardems** <https://github.com/ComputationalPhysiology/simcardems> — the
  cleanest modern EP + mechanics coupling reference.
- **Chaste** <https://github.com/Chaste/Chaste> — cardiac electromechanics tutorial
  with the full equation set.
