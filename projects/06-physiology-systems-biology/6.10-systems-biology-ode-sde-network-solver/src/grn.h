// ===========================================================================
// src/grn.h  --  Shared (host + device) gene-regulatory-network model + solvers
// ---------------------------------------------------------------------------
// Project 6.10 : Systems-Biology ODE/SDE Network Solver
//
// WHAT THIS PROJECT COMPUTES
//   An ENSEMBLE of gene-regulatory-network (GRN) trajectories: the SAME system
//   of coupled nonlinear ODEs solved for thousands of different parameter sets
//   (a parameter sweep). Each parameter set is an INDEPENDENT initial-value ODE
//   solve, so each GPU thread integrates one trajectory -- the natural mapping
//   for parameter sweeps / uncertainty quantification (UQ) / multi-cell studies.
//   This is exactly the batch-ODE pattern that SUNDIALS/CVODE-GPU and
//   libRoadRunner target (see ../README.md "Prior art").
//
// THE NETWORK: the "repressilator" (Elowitz & Leibler, Nature 2000)
//   A synthetic 3-gene ring where each gene REPRESSES the next:
//       gene0 -| gene1 -| gene2 -| gene0   (a cyclic negative-feedback loop)
//   With enough cooperativity the loop cannot settle and instead OSCILLATES --
//   the first engineered genetic clock. The state is 6 numbers: three mRNA
//   concentrations m0,m1,m2 and three protein concentrations p0,p1,p2.
//
//   Dimensionless model (one common non-dimensionalisation; see THEORY.md §math):
//       dm_i/dt = -m_i + alpha / (1 + p_{i-1}^n) + alpha0        (transcription)
//       dp_i/dt = -beta * (p_i - m_i)                            (translation)
//   with indices mod 3 (p_{-1} == p_2), and
//       alpha  = max transcription rate (leak-subtracted), in units of the
//                mRNA decay rate (time is measured in mRNA lifetimes),
//       alpha0 = leaky basal transcription (promoter never fully off),
//       n      = Hill coefficient (repression cooperativity; higher = sharper),
//       beta   = ratio of protein decay rate to mRNA decay rate.
//   The famous result: the fixed point loses stability (-> sustained
//   oscillations) when n and alpha are large and beta is moderate. We SWEEP
//   (alpha, n) across the ensemble and detect, per member, whether it oscillates.
//
//   THE DETERMINISTIC ODE ABOVE IS WHAT WE VERIFY (GPU vs CPU, RK4, see below).
//   A stochastic Chemical-Langevin (SDE) variant is provided too for teaching
//   (grn_cle_step) but is NOT part of the verified path -- an SDE has a random
//   term and so is not bit-reproducible across CPU/GPU RNG streams; THEORY.md
//   §numerics explains why, and README "Exercises" invites you to explore it.
//
//   The derivative AND the RK4 step live here as __host__ __device__ inline
//   functions so the CPU reference (reference_cpu.cpp) and the GPU kernel
//   (kernels.cu) integrate the SAME arithmetic -> their results match to
//   round-off. GRN_HD expands to `__host__ __device__` under nvcc and to nothing
//   under the plain host compiler (PATTERNS.md §2, the CPU/GPU-parity idiom).
//
// READ THIS AFTER: util/cuda_check.cuh; BEFORE: reference_cpu.h, kernels.cuh.
// ===========================================================================
#pragma once

#include <cmath>   // std::pow, std::fabs (host); nvcc maps these to device intrinsics

// ---- CPU/GPU-parity macro (PATTERNS.md §2) --------------------------------
// Under nvcc (__CUDACC__ defined) every function below is compiled for BOTH the
// host and the device from one source, guaranteeing identical math. Under the
// plain host compiler the decorators simply vanish. Keep CUDA-only constructs
// (e.g. __global__, blockIdx) OUT of this header so cl.exe/g++ can include it.
#ifdef __CUDACC__
#define GRN_HD __host__ __device__
// `#pragma unroll` is an nvcc directive. Emitting it from the plain host
// compiler (cl.exe, when it compiles reference_cpu.cpp) triggers warning C4068
// "unknown pragma", which this repo treats as a defect. GRN_UNROLL expands to
// the pragma under nvcc and to nothing on the host, so the small fixed-length
// state loops are unrolled on-device and stay warning-free on-host. (Using
// _Pragma lets us hide the pragma behind a macro.)
#define GRN_UNROLL _Pragma("unroll")
#else
#define GRN_HD
#define GRN_UNROLL
#endif

// Number of genes in the repressilator ring, and the resulting state size.
// Kept as compile-time constants so the state lives in registers/local arrays
// (no dynamic allocation inside the hot integration loop).
static constexpr int NUM_GENES = 3;              // g0 -| g1 -| g2 -| g0
static constexpr int STATE_DIM = 2 * NUM_GENES;  // [m0 m1 m2 p0 p1 p2] = 6

// One member's parameter set. Fixed fields (alpha0, beta, ICs, dt, steps) come
// from the config; the swept fields (alpha, n) are computed per member index.
struct GrnParams {
    double alpha;    // max transcription rate (leak-subtracted), swept
    double alpha0;   // leaky/basal transcription (promoter floor)
    double beta;     // protein-decay / mRNA-decay ratio
    double n;        // Hill coefficient (repression cooperativity), swept
};

// ---------------------------------------------------------------------------
// hill_repress: the repression response f(p) = 1 / (1 + p^n).
//   This is the fraction of promoter activity remaining when the upstream
//   repressor protein is at concentration p. It falls from 1 (p=0, fully ON)
//   toward 0 (large p, fully OFF); larger n makes the switch sharper. We clamp
//   p to be non-negative before the power so a tiny negative round-off excursion
//   can never feed a NaN into std::pow (pow of a negative base with a
//   non-integer exponent is undefined).
// ---------------------------------------------------------------------------
GRN_HD inline double hill_repress(double p, double n) {
    const double pp = (p > 0.0) ? p : 0.0;   // guard std::pow domain
    return 1.0 / (1.0 + pow(pp, n));         // pow() resolves to the host libm or the device intrinsic
}

// ---------------------------------------------------------------------------
// grn_deriv: the repressilator right-hand side.
//   Given the 6-component state s = [m0 m1 m2 p0 p1 p2] and the parameters,
//   write the 6 time-derivatives into d[]. Index arithmetic encodes the ring:
//   gene i is repressed by protein (i-1) mod 3, i.e. p[(i+2)%3].
//   s and d are plain C arrays of length STATE_DIM so this works identically in
//   host loops and in a single GPU thread's registers (no std::vector on-device).
// ---------------------------------------------------------------------------
GRN_HD inline void grn_deriv(const double* s, const GrnParams& pr, double* d) {
    // Split the flat state into the mRNA block [0..2] and protein block [3..5].
    const double* m = s;                 // m[i] = s[i]
    const double* p = s + NUM_GENES;     // p[i] = s[NUM_GENES + i]
    double* dm = d;                      // dm[i] -> d[i]
    double* dp = d + NUM_GENES;          // dp[i] -> d[NUM_GENES + i]

    GRN_UNROLL
    for (int i = 0; i < NUM_GENES; ++i) {
        const int rep = (i + NUM_GENES - 1) % NUM_GENES;   // upstream repressor gene = (i-1) mod 3
        // Transcription: mRNA decays (-m_i) and is produced at a Hill-repressed
        // rate plus a small constitutive leak alpha0.
        dm[i] = -m[i] + pr.alpha * hill_repress(p[rep], pr.n) + pr.alpha0;
        // Translation + protein decay: proteins relax toward their mRNA level at
        // relative rate beta.  dp_i/dt = -beta*(p_i - m_i).
        dp[i] = -pr.beta * (p[i] - m[i]);
    }
}

// ---------------------------------------------------------------------------
// grn_rk4_step: one classical 4th-order Runge-Kutta step of size dt (in place).
//   RK4 evaluates the derivative at four staged points and forms a weighted
//   average; its local error is O(dt^5) / global error O(dt^4). For the smooth,
//   mildly-stiff repressilator this is accurate and stable at the dt we use, and
//   -- crucially for teaching -- it is EXPLICIT, so the exact same closed-form
//   arithmetic runs on CPU and GPU (no linear solve, unlike an implicit BDF).
//   THEORY.md §algorithm discusses when you would switch to an implicit/CVODE
//   integrator for genuinely stiff metabolic models.
//
//   All temporaries are fixed-size stack arrays (STATE_DIM=6) so a GPU thread
//   keeps the entire working set in registers/local memory.
// ---------------------------------------------------------------------------
GRN_HD inline void grn_rk4_step(double* s, const GrnParams& pr, double dt) {
    double k1[STATE_DIM], k2[STATE_DIM], k3[STATE_DIM], k4[STATE_DIM];
    double tmp[STATE_DIM];

    grn_deriv(s, pr, k1);                                  // k1 = f(s)
    GRN_UNROLL
    for (int j = 0; j < STATE_DIM; ++j) tmp[j] = s[j] + 0.5 * dt * k1[j];
    grn_deriv(tmp, pr, k2);                                // k2 = f(s + dt/2 k1)
    GRN_UNROLL
    for (int j = 0; j < STATE_DIM; ++j) tmp[j] = s[j] + 0.5 * dt * k2[j];
    grn_deriv(tmp, pr, k3);                                // k3 = f(s + dt/2 k2)
    GRN_UNROLL
    for (int j = 0; j < STATE_DIM; ++j) tmp[j] = s[j] + dt * k3[j];
    grn_deriv(tmp, pr, k4);                                // k4 = f(s + dt   k3)

    // Weighted average: s += dt/6 (k1 + 2 k2 + 2 k3 + k4).
    GRN_UNROLL
    for (int j = 0; j < STATE_DIM; ++j)
        s[j] += (dt / 6.0) * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j]);
}

// ---------------------------------------------------------------------------
// grn_cle_step: ONE Chemical-Langevin (SDE) Euler-Maruyama step -- FOR TEACHING.
//   The Chemical Langevin Equation adds a noise term to each reaction flux to
//   model intrinsic molecular noise at moderate copy numbers:
//       dx = drift(x) dt + sum_r stoich_r * sqrt(a_r(x)) dW_r
//   Here `gauss[]` supplies one N(0,1) draw per state component (a lumped
//   diffusion approximation adequate for a teaching demo), `omega` is a system-
//   size scaling (larger volume -> relatively less noise, ~1/sqrt(omega)).
//   This function is NOT on the verified path (RNG streams differ CPU vs GPU);
//   it exists so the reader can see how one SDE step differs from the RK4 ODE
//   step -- exactly one added sqrt-flux*noise term. See README "Exercises".
// ---------------------------------------------------------------------------
GRN_HD inline void grn_cle_step(double* s, const GrnParams& pr, double dt,
                                double omega, const double* gauss) {
    double drift[STATE_DIM];
    grn_deriv(s, pr, drift);                               // deterministic part
    const double sqrt_dt = sqrt(dt);
    GRN_UNROLL
    for (int j = 0; j < STATE_DIM; ++j) {
        // Diffusion magnitude ~ sqrt(|flux| / omega): shot noise shrinks as the
        // system size (molecule count) omega grows.
        const double diff = sqrt(fabs(drift[j]) / omega);
        s[j] += drift[j] * dt + diff * sqrt_dt * gauss[j];
        if (s[j] < 0.0) s[j] = 0.0;                        // concentrations stay >= 0
    }
}

// ---------------------------------------------------------------------------
// Per-trajectory summary: the observables the ensemble analysis reports. We
// track the last gene's protein p2 as the "readout" (any gene works by
// symmetry). All fields are computed the SAME way on CPU and GPU.
// ---------------------------------------------------------------------------
struct MemberResult {
    double p2_final;    // protein-2 concentration at the final time
    double p2_min;      // minimum of p2 over the second half of the run
    double p2_max;      // maximum of p2 over the second half of the run
    int    zero_cross;  // # of upward level-crossings of p2 (2 per oscillation)
    int    oscillates;  // 1 if the member sustains oscillations, else 0
};

// ---------------------------------------------------------------------------
// integrate_member: integrate ONE ensemble member to completion and summarise.
//   Shared verbatim by the CPU reference loop and the GPU kernel (one thread),
//   so the summary matches to round-off. It uses a deliberate TWO-PASS design
//   for a subtle but important reason (see THEORY.md §verification):
//
//   PASS 1 -- integrate the full trajectory, and over the SECOND HALF (after
//     transients have decayed) record min, max, and the final p2. From these we
//     form a FIXED reference level `mid = (min+max)/2` and amplitude `amp`.
//
//   PASS 2 -- re-integrate and count upward crossings of `mid` using HYSTERESIS
//     with a deadband of 25% of the amplitude: the signal must fall clearly
//     below (mid - band) and then rise clearly above (mid + band) to score one
//     crossing. Two crossings = one full oscillation.
//
//   Why two passes + hysteresis + a FIXED level (not a running mean): a genuine
//   steady state is "flat" only to round-off, and a running-mean crossing test
//   would count spurious crossings from those round-off wiggles -- and, worse,
//   the COUNT would differ between an -O2 (FMA-contracted) Release build and an
//   unoptimised Debug build, breaking the "stdout identical across builds"
//   contract. The deadband is proportional to the amplitude, so on a flat member
//   `band` is ~round-off and the signal never travels the full band -> exactly 0
//   crossings on EVERY build. This makes zero_cross a deterministic observable.
//
//   Cost: two RK4 passes (2x work) -- negligible for a teaching demo, and a nice
//   illustration that a numerically ROBUST feature detector is worth a constant
//   factor. (An exercise in the README replaces this with an FFT-based detector.)
// ---------------------------------------------------------------------------
GRN_HD inline MemberResult integrate_member(const double* s0, const GrnParams& pr,
                                            double dt, int steps) {
    const int P2 = NUM_GENES + (NUM_GENES - 1);   // index of protein p2 in the state
    const int half = steps / 2;                   // start observing after transients

    // ---- Pass 1: extrema + final value over the second half ---------------
    double s[STATE_DIM];
    GRN_UNROLL
    for (int j = 0; j < STATE_DIM; ++j) s[j] = s0[j];
    double vmin = 1e300, vmax = -1e300;
    for (int step = 1; step <= steps; ++step) {
        grn_rk4_step(s, pr, dt);
        if (step > half) {
            const double v = s[P2];
            if (v < vmin) vmin = v;
            if (v > vmax) vmax = v;
        }
    }
    const double p2_final = s[P2];
    const double amp = vmax - vmin;
    const double mid = 0.5 * (vmax + vmin);
    const double band = 0.25 * amp;               // hysteresis deadband (round-off-safe on flat signals)
    const double lo = mid - band, hi = mid + band;

    // ---- Pass 2: hysteretic upward-crossing count of the FIXED level `mid` --
    GRN_UNROLL
    for (int j = 0; j < STATE_DIM; ++j) s[j] = s0[j];
    int crossings = 0;
    int state = 0;   // -1 = last seen below `lo`, +1 = last seen above `hi`, 0 = undetermined
    for (int step = 1; step <= steps; ++step) {
        grn_rk4_step(s, pr, dt);
        if (step > half) {
            const double v = s[P2];
            if (v > hi) {
                if (state == -1) ++crossings;      // completed a low->high excursion
                state = 1;
            } else if (v < lo) {
                state = -1;
            }
            // v inside [lo,hi]: within the deadband, ignore (that is the point).
        }
    }

    MemberResult r;
    r.p2_final   = p2_final;
    r.p2_min     = vmin;
    r.p2_max     = vmax;
    r.zero_cross = crossings;
    // "Oscillates" if the relative swing is appreciable AND we saw >= 2 upward
    // crossings (i.e. at least a couple of full cycles). A steady state has
    // amp/mid ~ 0 and 0 crossings, so it is correctly classified as non-oscillatory.
    const bool big_swing = (mid > 1e-9) ? (amp / mid > 0.05) : (amp > 1e-6);
    r.oscillates = (big_swing && crossings >= 2) ? 1 : 0;
    return r;
}
