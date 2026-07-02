# THEORY — 6.23 Glucose-Insulin Dynamics & Artificial Pancreas

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not a medical device, not for any clinical decision._

---

## 1. The science

People with **type 1 diabetes (T1D)** make little or no insulin, the hormone that
lets cells take up glucose from blood. Without it, blood glucose swings between
dangerous highs (**hyperglycemia**, > 180 mg/dL, long-term organ damage) and lows
(**hypoglycemia**, < 70 mg/dL, acutely life-threatening). Management means
delivering the right insulin at the right time — historically by hand, now
increasingly by an **artificial pancreas**: a continuous glucose monitor (CGM) +
an insulin pump + a control algorithm that closes the loop automatically.

Designing and testing such a controller on real patients is slow and risky, so
the field relies on **in-silico trials**: simulate a whole *cohort* of virtual
patients with a validated glucose-insulin model, run the candidate controller on
all of them, and measure safety and efficacy before any human is involved. The
**UVA/Padova T1D simulator** is so trusted that the FDA accepted it as a
substitute for animal trials in 2008. The same simulation loop is the environment
in which **reinforcement-learning** insulin-dosing policies are trained — millions
of simulated patient-hours.

The core question this project answers: *given a cohort of patients who differ in
physiology, how well does one closed-loop controller keep each of them in range
after a meal, and where does it risk hypoglycemia?* Answering it for a large
cohort is exactly the parallel workload the GPU is built for.

## 2. The math

We use the **Bergman "minimal model"** — the teaching workhorse of glucose
kinetics — with three state variables:

- `G(t)` — plasma glucose concentration [mg/dL]
- `X(t)` — *remote* insulin action in the interstitium [1/min] (the insulin that
  actually drives glucose uptake; it lags plasma insulin, hence its own compartment)
- `I(t)` — plasma insulin concentration [µU/mL]

Governing ODEs:

```
dG/dt = -(p1 + X)·G + p1·Gb + Ra(t)/VG
dX/dt = -p2·X + p3·(I - Ib)
dI/dt = -n·(I - Ib) + u(t)/VI
```

with parameters (units in `src/bergman.h`):

- `p1` = **glucose effectiveness** `SG`: glucose self-clearance at basal insulin.
- `p2`, `p3` = insulin-action kinetics. The steady remote action for a given
  plasma insulin is `X* = (p3/p2)·(I − Ib)`, so **`SI = p3/p2` is the insulin
  sensitivity** — the single most patient-variable parameter (low `SI` = insulin
  resistant).
- `n` = plasma-insulin clearance; `Gb, Ib` = basal set-points; `VG, VI` =
  distribution volumes that scale the meal and pump inputs.

**Meal disturbance** `Ra(t)` (glucose entering plasma from the gut) uses the
two-exponential *gastric-emptying* impulse response:

```
Ra(t) = D·Ag·k²·(t − t_meal)·exp(−k·(t − t_meal)),   t ≥ t_meal
```

`D` = carbs [mg], `Ag` = bioavailability, `k` = 1/absorption-time. It rises to a
single peak at `t_meal + 1/k` then decays; its integral is `D·Ag` (total glucose
absorbed).

**Controller** `u(t)` (the artificial pancreas) is a discrete **PID** that runs
every `control_dt` minutes:

```
e        = G − G_target
u = clamp( u_basal + Kp·e + Ki·∫e dt + Kd·dG/dt ,  0 ,  u_max )
```

Insulin can only be *added* (`u ≥ 0` — you cannot remove it), so the command is
clamped; a simple **anti-windup** rule stops the integral term ballooning while
the pump is saturated. The objective metrics are the standard trial outcomes:
**time-in-range** (TIR, fraction of time in [70, 180] mg/dL), hypoglycemia
fraction, mean glucose, and total insulin delivered.

## 3. The algorithm

Per patient, integrate the 3-state ODE with classical **4th-order Runge-Kutta
(RK4)** at fixed step `dt`, holding the controller command constant across each
step (**zero-order hold** — exactly how a digital controller behaves between
updates), and recomputing `u` at each control tick:

```
state = (G0, X=0, I=Ib);  u = u_basal
for k in 0 .. steps-1:
    if k is a control tick:  u = PID(state)     # zero-order hold until next tick
    RK4 step the ODE by dt under (u, Ra(t+dt/2))
    accumulate metrics (min/max/mean G, TIR, hypo, insulin)
```

**Complexity.** One patient costs `O(steps)` (RK4 = 4 derivative evaluations per
step). The cohort of `M` patients is `O(M·steps)`. Crucially, patients are
**independent** — no data is shared between them — so the *parallel depth* is just
`O(steps)` (one patient's serial time) while the *work* is `O(M·steps)`. That is
the textbook profile of an embarrassingly parallel ensemble.

**Arithmetic intensity** is high and memory traffic is tiny: each thread reads a
small parameter struct once and writes one result struct at the end; the entire
trajectory lives in registers. This is compute-bound, not bandwidth-bound — ideal
for the GPU.

## 4. The GPU mapping

**Pattern: ensemble ODE integration — one thread per virtual patient**
(docs/PATTERNS.md §1; the same pattern as flagship `9.02` SEIR and `13.02` PBPK).

- **Thread-to-data map:** `idx = blockIdx.x·blockDim.x + threadIdx.x` owns patient
  `idx`. It builds that patient's parameters from the cohort grid
  (`patient_params`), runs the *entire* RK4 + PID loop in local registers, and
  writes one `PatientResult` to global memory. No shared memory, no atomics, no
  inter-thread communication.
- **Launch configuration:** `block = 128` threads, `grid = ceil(M/128)`. We keep
  the block small because each thread is **register-heavy** — it holds the 3 ODE
  states, the controller memory, and 12 RK4 temporaries — so a smaller block
  leaves enough registers per thread for good occupancy on `sm_75…sm_89`.
- **Memory hierarchy:** the trajectory is entirely in **registers**; the only
  global-memory touch is the single result write. The `CohortConfig` is passed
  **by value** as a kernel argument (it rides in the constant argument bank), so
  there are no input copies at all.
- **Divergence:** mild — all patients run the same step count, so warps stay in
  lockstep through the loop; only the `min/max/in-range` branches differ per step,
  which is cheap predication.

```
grid:  [ block 0 ][ block 1 ] ... [ block ceil(M/128)-1 ]
block:  128 threads
         └─ thread t ── patient (blockIdx*128 + t)
              └─ registers: G, X, I, err_acc, G_prev, u, k1..k4  → RK4 loop
                   └─ one write: out[idx] = PatientResult
```

**Why not a library ODE solver?** The catalog mentions "cusolve / custom RK4
kernel". There is no drop-in batched non-stiff ODE integrator in the CUDA math
libraries; cuSOLVER solves *linear* systems, not IVPs. Hand-rolling RK4 is a few
lines, teaches the method, and — because it is shared with the CPU reference — is
what makes verification exact (see §6). cuRAND would enter if we sampled random
meal disturbances per patient (an exercise).

## 5. Numerical considerations

- **Precision: FP64 (double) throughout.** The dynamics are mildly stiff (insulin
  clearance `n ≈ 0.14`/min vs. slow glucose terms) and we integrate thousands of
  steps; single precision would accumulate visible drift. Double keeps the model
  faithful and makes CPU/GPU agreement tight.
- **Stability:** fixed-step RK4 is stable here because `dt = 0.5 min` is well below
  the fastest time constant (`1/n ≈ 7 min`). Larger `dt` eventually rings/blows up
  — a good exercise. The zero-order-hold controller is intentionally *not* updated
  mid-step, matching real digital control.
- **Determinism:** there are **no atomics and no reductions across threads** — each
  thread owns its patient end to end, so results are order-independent and
  bit-identical across runs. stdout is therefore reproducible (PATTERNS.md §3).
- **CPU vs GPU:** the *only* possible divergence is the GPU's fused multiply-add
  (FMA) contracting `a*b+c` differently from the host compiler. Over thousands of
  RK4 steps this shows up at ~`1e-13` on glucose — far below any clinical or even
  display resolution.

## 6. How we verify correctness

`src/reference_cpu.cpp` runs the *same* `simulate_patient()` from `src/bergman.h`
(a shared `__host__ __device__` function) serially over the cohort. The GPU kernel
calls the identical function from one thread per patient. `main.cu` compares the
headline per-patient metrics (min/max/mean glucose and TIR) and reports the worst
absolute difference.

- **Tolerance: `1e-4` mg/dL.** Chosen to sit comfortably above the FMA-induced
  double-precision divergence (~`1e-13` observed) yet absurdly below anything
  physiologically meaningful. We verify to a *physically negligible* tolerance and
  say so — we do **not** pretend the two are bit-identical (PATTERNS.md §4).
- **Why this is convincing:** an independent serial implementation and a parallel
  one, written to different execution models, agreeing to round-off is strong
  evidence neither has a logic bug.
- **A second, physical sanity check:** the results reproduce known physiology —
  higher insulin sensitivity lowers the peak and mean glucose and reduces the
  insulin needed, while pushing the glucose nadir toward the hypo boundary. That
  validates the *model*, not just CPU==GPU agreement.

## 7. Where this sits in the real world

This is a deliberately **reduced-scope teaching version**:

- **Model.** The FDA-accepted **UVA/Padova** simulator is a ~13-compartment
  nonlinear model (gut, liver endogenous production, subcutaneous insulin
  absorption, renal clearance, glucose transport) fitted to a large virtual
  population. We use the 3-state **Bergman minimal model**, which captures the same
  qualitative loop with far fewer parameters — perfect for teaching, not for
  device certification.
- **Sensing.** A real CGM is **noisy and delayed**; production controllers add a
  **Kalman filter** for state estimation. We feed the controller noiseless glucose
  (Exercise 3 adds cuRAND noise + a Kalman filter).
- **Controller.** We use a plain **PID**. Commercial systems use **model-predictive
  control (MPC)** or learned policies (**RL**: PPO/SAC). The reason to simulate a
  cohort on the GPU in the first place is to *train and validate* those — this
  project is the environment underneath, in miniature.
- **Scale.** Real RL/UQ studies run 10³–10⁶ patients across many episodes; our
  1024-patient sample is a legible slice of the same parallel workload.

Production references: **simglucose** (Python UVA/Padova + gym), **GluCoEnv**
(GPU RL env), **G2P2C** (RL artificial pancreas), **OpenAPS oref0** (open-source
reference dosing algorithm).

---

## References

- Bergman RN et al., *Quantitative estimation of insulin sensitivity* (Am J
  Physiol, 1979) — the minimal model this project implements.
- Man CD, Micheletto F, Lv D, Breton M, Kovatchev B, Cobelli C, *The UVA/PADOVA
  Type 1 Diabetes Simulator: New Features* (J Diabetes Sci Technol, 2014) — the
  FDA-accepted full model contrasted in §7.
- Hovorka R et al., *Nonlinear model predictive control of glucose* — MPC for the
  artificial pancreas.
- **simglucose** <https://github.com/jxx123/simglucose> — Python UVA/Padova + RL
  gym; study its patient parameters and meal/scenario handling.
- **GluCoEnv** <https://github.com/chirathyh/GluCoEnv> — how a GPU RL glucose-
  control environment batches patients (the production form of this project).
- **G2P2C** <https://github.com/RL4H/G2P2C> — an end-to-end RL artificial-pancreas
  agent; the policy that would replace our PID.
- **OpenAPS oref0** <https://github.com/openaps/oref0> — a real, deployed
  open-source dosing algorithm; read it to see production safety constraints.
