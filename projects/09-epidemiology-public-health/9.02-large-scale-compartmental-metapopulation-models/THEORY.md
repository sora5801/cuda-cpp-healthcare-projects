# THEORY — 9.02 Large-Scale Compartmental & Metapopulation Models

> For a reader who knows C++ but is new to CUDA and to epidemic modelling.
> See [README.md](README.md) for the tour and build. _Educational only._

## 1. The science

To anticipate an outbreak, epidemiologists divide a population into
**compartments** by infection status and write down how people flow between them.
The **SEIR** model has four: **S**usceptible, **E**xposed (infected but not yet
infectious), **I**nfectious, **R**ecovered. Because the true parameters are
uncertain, a single simulation is not enough — you run an **ensemble** over many
plausible parameter sets and study the distribution of outcomes (peak height,
timing, attack rate). That ensemble is the compute load this project parallelizes.

## 2. The math

With constant population `N = S+E+I+R`, the SEIR ODEs are

```
dS/dt = -β S I / N
dE/dt =  β S I / N - σ E
dI/dt =  σ E      - γ I
dR/dt =  γ I
```

`β` is the transmission rate, `1/σ` the latent period, `1/γ` the infectious
period. The **basic reproduction number** `R0 = β/γ` is the average number of
secondary infections from one case in a fully susceptible population: `R0 > 1`
produces an epidemic, `R0 < 1` fizzles. There is no closed-form `I(t)`, so we
integrate numerically.

## 3. The algorithm

**Runge-Kutta 4 (RK4)** advances the state by sampling the derivative at four
points per step and combining them:

```
k1 = f(y)              k2 = f(y + dt/2 k1)
k3 = f(y + dt/2 k2)    k4 = f(y + dt k3)
y += dt/6 (k1 + 2 k2 + 2 k3 + k4)         # O(dt^4) local error
```

Each ensemble member runs this loop for `steps` timesteps and records its peak
infectious fraction, the day of the peak, and the final `R/N` (attack rate).

**Complexity.** Each member is `O(steps)` sequential RK4 steps; the ensemble is
`M` members `×` that. The members are independent — perfect parallelism.

## 4. The GPU mapping

**Decomposition.** One thread per ensemble member. Thread `idx` reads its
`(β, γ)` from the sweep grid (`member_params`), then runs the **entire RK4 time
loop in registers/local memory** and writes a single `MemberResult`. There is no
inter-thread communication and no global memory traffic during integration — the
state lives in registers — so it is bandwidth-light and compute-bound, exactly the
regime where a GPU's many cores shine.

```
  member 0 (beta0,gamma0):  RK4 ----------------> result[0]
  member 1 (beta1,gamma1):  RK4 ----------------> result[1]
  ...                                              (one thread each, in parallel)
  member M-1:               RK4 ----------------> result[M-1]
```

**Divergence.** All members run the same number of steps, so the only divergence
is the peak-tracking branch — negligible. (Adaptive step sizes, Exercise 2, would
introduce real divergence, since different members would take different numbers of
steps.)

**Precision.** We integrate in **double**: epidemic curves span many orders of
magnitude (from `I0=10` to millions) over hundreds of steps, and double precision
keeps the CPU and GPU in lock-step. Because the shared `__host__ __device__`
integrator runs the identical operations, the per-member results match to ~`1e-15`
— round-off, not algorithm.

## 5. Numerical considerations

- **Determinism.** No reductions, no atomics during integration → reproducible and
  CPU-matching. (The ensemble summary stats are computed afterward on the host.)
- **Stability.** RK4 is stable for these non-stiff dynamics at the chosen `dt`;
  very large `β` or tiny `1/γ` could require a smaller step (or an implicit/
  adaptive method — Exercise 2).
- **Conservation.** `S+E+I+R = N` should hold; RK4 conserves it to round-off here.

## 6. How we verify correctness

`main.cu` integrates the whole ensemble twice — `integrate_cpu` (a serial loop)
and `integrate_gpu` (one thread per member) — and compares the per-member peak
infection and attack rate (`worst diff ≈ 1e-15`). Beyond CPU/GPU parity, the
results are epidemiologically sensible: members with larger `R0 = β/γ` peak
earlier and higher and infect a larger fraction; `R0 < 1` members never take off.
That is the model behaving correctly, not just two codes agreeing.

## 7. Where this sits in the real world

Production epidemic models add: **metapopulation structure** (many geographic
patches coupled by a mobility matrix — the per-step update becomes a batched
**sparse matrix-vector multiply**, a cuSPARSE operation), **age structure** and
contact matrices, **seasonal forcing**, demographic and observational
**stochasticity** (SDEs), and **adaptive/stiff solvers** (`dopri5` on GPU via
Torchdiffeq). The ensemble-over-threads pattern you learn here is the backbone of
the uncertainty-quantification step in all of them.

## References

- Kermack & McKendrick (1927) — the original compartmental epidemic model.
- Keeling & Rohani, *Modeling Infectious Diseases* — the standard text.
- Press et al., *Numerical Recipes* — Runge-Kutta integration.
- NVIDIA cuSPARSE documentation — batched sparse mat-vec for metapopulation coupling.
