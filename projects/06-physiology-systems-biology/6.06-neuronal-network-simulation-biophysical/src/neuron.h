// ===========================================================================
// src/neuron.h  --  Shared (host + device) biophysical neuron model
// ---------------------------------------------------------------------------
// Project 6.6 : Neuronal Network Simulation (Biophysical)
//
// WHAT THIS PROJECT COMPUTES
//   A NETWORK of morphologically-detailed neurons. Each neuron is a
//   MULTI-COMPARTMENT cable: a soma plus a short chain of dendritic segments,
//   every segment carrying its own Hodgkin-Huxley (HH) membrane. The membrane
//   voltage of a compartment obeys the CABLE EQUATION -- current diffuses to its
//   neighbours through the axial (intracellular) resistance -- so the segments
//   are COUPLED and must be solved together each timestep. The neurons in the
//   network are coupled to EACH OTHER through chemical synapses: when a neuron's
//   soma crosses threshold ("spikes"), it delivers an excitatory conductance
//   pulse to its postsynaptic partners on the next step.
//
//   This single header holds every piece of per-neuron physics as
//   `__host__ __device__` inline functions, so the CPU reference
//   (reference_cpu.cpp) and the GPU kernel (kernels.cu) integrate the SAME
//   arithmetic and their results match to round-off. NN_HD expands to
//   __host__ __device__ under nvcc and to nothing under the host compiler
//   (the "HD-macro idiom", docs/PATTERNS.md section 2).
//
//   Keep this header free of CUDA-only constructs (no __global__, no
//   <cuda_runtime.h>) so the plain host compiler can include it too.
//
// THE THREE NUMERICAL PIECES (all standard compartmental-modelling technique)
//   1. HH GATING (m,h,n) advanced by the RUSH-LARSEN rule: treat each gate's
//      ODE  dx/dt = (x_inf - x)/tau_x  as locally linear over dt and use its
//      EXACT exponential solution  x <- x_inf + (x - x_inf)*exp(-dt/tau_x).
//      Unconditionally stable -- gates never overshoot [0,1] -- which is why
//      real simulators (NEURON) use it instead of forward Euler.
//   2. CABLE COUPLING solved IMPLICITLY (backward Euler) each step. For an
//      UNBRANCHED cable the resulting linear system is TRIDIAGONAL, and the
//      HINES solver reduces to the classic Thomas algorithm (one forward sweep
//      to eliminate the sub-diagonal, one back-substitution sweep). O(ncomp)
//      work, unconditionally stable in dt.
//   3. SYNAPSES modelled as an exponentially-decaying excitatory conductance
//      g_syn (AMPA-like): a presynaptic spike ADDS a quantum to g_syn, which
//      then decays as g_syn <- g_syn*exp(-dt/tau_syn) and drives a current
//      g_syn*(E_syn - V) into the soma. Event-driven: the "event" is simply the
//      presynaptic spike flag from the previous step (see kernels.cu / main.cu).
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh.  READ BEFORE: kernels.cu.
// ===========================================================================
#pragma once

#include <cmath>   // std::exp, std::fabs  (host); nvcc maps these to device intrinsics

// HD-macro: __host__ __device__ under nvcc, nothing under the host compiler.
#ifdef __CUDACC__
#define NN_HD __host__ __device__
#else
#define NN_HD
#endif

// ---------------------------------------------------------------------------
// Compile-time model sizes. Keeping them constants (not runtime values) lets
// every neuron's state live in fixed-size register/local arrays inside a single
// thread -- no dynamic allocation on the device, which is what makes the
// one-thread-per-neuron mapping clean and fast.
// ---------------------------------------------------------------------------
#ifndef NN_MAX_COMP
#define NN_MAX_COMP 8      // compartments per neuron (compartment 0 = soma)
#endif

// ---------------------------------------------------------------------------
// HHParams: the fixed biophysical constants shared by every compartment.
//   Units follow the classic single-compartment HH convention:
//     - voltages in millivolts (mV), measured relative to a resting-ish datum
//     - conductances (g_*) in millisiemens per cm^2 (mS/cm^2)
//     - capacitance cm in microfarads per cm^2 (uF/cm^2)
//     - time in milliseconds (ms)
//   With those units the current density I = g*(E - V) is in uA/cm^2 and
//   dV/dt = I/cm comes out in mV/ms -- a self-consistent set.
// ---------------------------------------------------------------------------
struct HHParams {
    double cm      = 1.0;     // membrane capacitance                (uF/cm^2)
    double gNa     = 120.0;   // max sodium conductance              (mS/cm^2)
    double gK      = 36.0;    // max potassium conductance           (mS/cm^2)
    double gL      = 0.3;     // leak conductance                    (mS/cm^2)
    double eNa     = 50.0;    // sodium reversal potential           (mV)
    double eK      = -77.0;   // potassium reversal potential        (mV)
    double eL      = -54.387; // leak reversal potential             (mV)
    double gAxial  = 0.10;    // inter-compartment (axial) coupling  (mS/cm^2)
                              //   couples neighbouring segments; larger =
                              //   stiffer cable (voltage equalises faster).
    double eSyn    = 0.0;     // excitatory synaptic reversal (AMPA) (mV)
    double tauSyn  = 2.0;     // synaptic conductance decay time     (ms)
    double wSyn    = 0.30;    // conductance added per presyn spike  (mS/cm^2)
    double vThresh = 0.0;     // soma spike-detection threshold      (mV)
};

// ---------------------------------------------------------------------------
// HH steady-state / time-constant helper functions. Each returns the classic
// rate constant alpha_x(V) or beta_x(V) for a gate x in {m,h,n} at voltage V.
// These are the standard Hodgkin-Huxley (1952) squid-axon expressions (shifted
// so that rest is near -65 mV). We factor them out so the derivative code reads
// like the biology. The `(V + c)` singularities in the alpha_m / alpha_n forms
// are removed with the well-known l'Hopital limit (the 1e-6 guard below).
// ---------------------------------------------------------------------------

// Sodium activation gate m.
NN_HD inline double alpha_m(double V) {
    const double d = V + 40.0;
    // As V -> -40, both numerator and denominator -> 0; the limit is 1.0.
    if (std::fabs(d) < 1e-6) return 1.0;
    return 0.1 * d / (1.0 - std::exp(-d / 10.0));
}
NN_HD inline double beta_m(double V) {
    return 4.0 * std::exp(-(V + 65.0) / 18.0);
}

// Sodium inactivation gate h.
NN_HD inline double alpha_h(double V) {
    return 0.07 * std::exp(-(V + 65.0) / 20.0);
}
NN_HD inline double beta_h(double V) {
    return 1.0 / (1.0 + std::exp(-(V + 35.0) / 10.0));
}

// Potassium activation gate n.
NN_HD inline double alpha_n(double V) {
    const double d = V + 55.0;
    if (std::fabs(d) < 1e-6) return 0.1;   // l'Hopital limit at V -> -55
    return 0.01 * d / (1.0 - std::exp(-d / 10.0));
}
NN_HD inline double beta_n(double V) {
    return 0.125 * std::exp(-(V + 65.0) / 80.0);
}

// ---------------------------------------------------------------------------
// rush_larsen: advance ONE gating variable by dt using the Rush-Larsen rule.
//   Given alpha, beta at the current voltage:
//       x_inf = alpha/(alpha+beta),  tau = 1/(alpha+beta)
//   and the ODE dx/dt = (x_inf - x)/tau has the EXACT solution over dt
//       x(t+dt) = x_inf + (x - x_inf) * exp(-dt/tau).
//   Using the exact solution (rather than Euler) keeps x in [0,1] for any dt,
//   which is why it is the workhorse gate integrator in production simulators.
//   Returns the updated gate value; `x` is passed by value (pure function).
// ---------------------------------------------------------------------------
NN_HD inline double rush_larsen(double x, double a, double b, double dt) {
    const double sum = a + b;
    // sum is a+b of two positive rate constants, so it is > 0 in practice; the
    // guard just avoids a divide-by-zero if both rates underflow to 0.
    if (sum < 1e-12) return x;
    const double x_inf = a / sum;          // steady-state open fraction
    const double tau   = 1.0 / sum;        // time constant (ms)
    return x_inf + (x - x_inf) * std::exp(-dt / tau);
}

// ---------------------------------------------------------------------------
// NeuronState: the full dynamical state of ONE neuron, stored in fixed-size
// arrays so it fits in a single thread's local memory (no heap on the device).
//   Index 0 is the SOMA; indices 1..ncomp-1 are dendritic compartments in a
//   line (an unbranched cable). g_syn is a single soma-directed synaptic
//   conductance (the network's excitatory input to this cell).
// ---------------------------------------------------------------------------
struct NeuronState {
    double V[NN_MAX_COMP];   // membrane voltage per compartment (mV)
    double m[NN_MAX_COMP];   // Na activation gate   (dimensionless, [0,1])
    double h[NN_MAX_COMP];   // Na inactivation gate (dimensionless, [0,1])
    double n[NN_MAX_COMP];   // K  activation gate   (dimensionless, [0,1])
    double g_syn;            // excitatory synaptic conductance at soma (mS/cm^2)
    int    ncomp;            // number of active compartments (<= NN_MAX_COMP)
};

// ---------------------------------------------------------------------------
// init_neuron: put a neuron at its resting state.
//   Voltages start at v_rest; each gate starts at its steady-state value for
//   that voltage (x_inf = alpha/(alpha+beta)) so the cell is genuinely at rest
//   and will not twitch spuriously on step 1.
// ---------------------------------------------------------------------------
NN_HD inline void init_neuron(NeuronState& s, int ncomp, double v_rest) {
    s.ncomp = ncomp;
    s.g_syn = 0.0;
    for (int c = 0; c < ncomp; ++c) {
        s.V[c] = v_rest;
        s.m[c] = alpha_m(v_rest) / (alpha_m(v_rest) + beta_m(v_rest));
        s.h[c] = alpha_h(v_rest) / (alpha_h(v_rest) + beta_h(v_rest));
        s.n[c] = alpha_n(v_rest) / (alpha_n(v_rest) + beta_n(v_rest));
    }
}

// ---------------------------------------------------------------------------
// ionic_current: the total ACTIVE membrane current density leaving compartment
// `c` at its present voltage/gates (Na + K + leak), in uA/cm^2, written in the
// "outward positive" convention actually, we return I_ion such that
//   cm dV/dt = -I_ion + (axial + synaptic) currents.
// Here we return the HH ionic term  gNa*m^3*h*(V-eNa) + gK*n^4*(V-eK)
//                                     + gL*(V-eL).
// (This is the quantity that, moved to the RHS with a minus sign, drives V.)
// ---------------------------------------------------------------------------
NN_HD inline double ionic_current(const NeuronState& s, int c, const HHParams& p) {
    const double V = s.V[c];
    const double m3h = s.m[c] * s.m[c] * s.m[c] * s.h[c];   // m^3 h
    const double n4  = s.n[c] * s.n[c] * s.n[c] * s.n[c];   // n^4
    const double iNa = p.gNa * m3h * (V - p.eNa);
    const double iK  = p.gK  * n4  * (V - p.eK);
    const double iL  = p.gL        * (V - p.eL);
    return iNa + iK + iL;
}

// ---------------------------------------------------------------------------
// thomas_solve: solve a tridiagonal linear system  a[i]*x[i-1] + b[i]*x[i] +
//   c[i]*x[i+1] = d[i]  (i = 0..N-1, with a[0]=c[N-1]=0) IN PLACE, returning the
//   solution in d[]. This is the HINES SOLVER specialised to an unbranched cable
//   -- for a branching dendritic tree Hines orders the compartments so the same
//   two-sweep elimination still works; on a line it IS the Thomas algorithm.
//
//   Forward sweep: eliminate each sub-diagonal entry a[i], carrying modified
//   c'[i] and d'[i]. Back substitution: solve from the last row upward.
//   Work is O(N), and because our system is diagonally dominant (backward-Euler
//   of a diffusion operator) the elimination is numerically stable with NO
//   pivoting -- exactly why compartmental simulators can afford it every step.
//
//   Arrays are length N (= ncomp). cc[] is scratch for the modified super-
//   diagonal; we keep it separate so a[],b[],c[] stay read-only-ish and the code
//   maps 1:1 onto the textbook algorithm.
// ---------------------------------------------------------------------------
NN_HD inline void thomas_solve(const double* a, const double* b, const double* c,
                               double* d, double* cc, int N) {
    // Row 0: no sub-diagonal. c'[0] = c[0]/b[0], d'[0] = d[0]/b[0].
    cc[0] = c[0] / b[0];
    d[0]  = d[0] / b[0];
    // Forward elimination of the sub-diagonal.
    for (int i = 1; i < N; ++i) {
        const double denom = b[i] - a[i] * cc[i - 1];   // pivot after eliminating a[i]
        cc[i] = c[i] / denom;
        d[i]  = (d[i] - a[i] * d[i - 1]) / denom;
    }
    // Back substitution: x[N-1] already sits in d[N-1]; propagate upward.
    for (int i = N - 2; i >= 0; --i) {
        d[i] = d[i] - cc[i] * d[i + 1];   // d[i] - cc[i]*x[i+1]
    }
}

// ---------------------------------------------------------------------------
// step_neuron: advance ONE neuron by a single timestep dt.
//   Order of operations (staggered, as in NEURON):
//     (1) Update every gate with Rush-Larsen at the CURRENT voltages.
//     (2) Decay the synaptic conductance and add any incoming spike quantum.
//     (3) Solve the IMPLICIT (backward-Euler) cable system for the new voltages
//         with the Hines/Thomas tridiagonal solver.
//     (4) Detect a soma spike (upward threshold crossing) and report it.
//
//   Backward-Euler cable system, per compartment c, unknown = V_new[c]:
//       cm/dt * V_new[c]
//         = cm/dt * V_old[c]                       (capacitive memory)
//           - I_ion(V_old, gates_new)              (HH ionic, evaluated at V_old)
//           + gAxial*(V_new[left]  - V_new[c])     (axial coupling, IMPLICIT)
//           + gAxial*(V_new[right] - V_new[c])
//           + [c==0] * g_syn*(eSyn - V_new[0])     (synapse, implicit in V_new)
//   Moving the V_new terms to the left gives a tridiagonal system:
//       a[c] = -gAxial                 (coupling to left neighbour)
//       c[c] = -gAxial                 (coupling to right neighbour)
//       b[c] =  cm/dt + (#neighbours)*gAxial  (+ g_syn on the soma row)
//       d[c] =  cm/dt*V_old[c] - I_ion[c]      (+ g_syn*eSyn on the soma row)
//   We treat the HH ionic term explicitly (evaluated at V_old) -- a standard,
//   accurate choice for small dt -- while the linear diffusion + synapse are
//   implicit, which is what buys the unconditional stability of the coupling.
//
//   `spike_in` is 1 if a presynaptic partner fired on the PREVIOUS step (the
//   event-driven synaptic input). Returns 1 if THIS neuron's soma spikes now,
//   else 0. `v_prev_soma` lets the caller see the pre-update soma voltage for a
//   clean rising-edge threshold test.
// ---------------------------------------------------------------------------
NN_HD inline int step_neuron(NeuronState& s, const HHParams& p, double dt, int spike_in) {
    const int N = s.ncomp;
    const double v_soma_old = s.V[0];   // for the rising-edge spike test

    // (1) Gates: Rush-Larsen update at the current (old) voltages. We snapshot
    //     the ionic current AFTER updating gates but using the OLD voltage --
    //     this is the conventional NEURON ordering and keeps CPU==GPU identical.
    for (int c = 0; c < N; ++c) {
        const double V = s.V[c];
        s.m[c] = rush_larsen(s.m[c], alpha_m(V), beta_m(V), dt);
        s.h[c] = rush_larsen(s.h[c], alpha_h(V), beta_h(V), dt);
        s.n[c] = rush_larsen(s.n[c], alpha_n(V), beta_n(V), dt);
    }

    // (2) Synapse: exponential decay of the conductance, then add this step's
    //     incoming quantum if a presynaptic partner spiked. Event-driven: the
    //     "event" is the spike_in flag (from the previous step's spike buffer).
    s.g_syn *= std::exp(-dt / p.tauSyn);
    if (spike_in) s.g_syn += p.wSyn;

    // (3) Assemble and solve the tridiagonal cable system (Hines/Thomas).
    double a[NN_MAX_COMP], b[NN_MAX_COMP], csup[NN_MAX_COMP];
    double d[NN_MAX_COMP], cc[NN_MAX_COMP];
    const double cm_dt = p.cm / dt;
    for (int c = 0; c < N; ++c) {
        int neighbours = 0;
        a[c] = 0.0; csup[c] = 0.0;
        if (c > 0)     { a[c]    = -p.gAxial; ++neighbours; }  // couple to left
        if (c < N - 1) { csup[c] = -p.gAxial; ++neighbours; }  // couple to right
        b[c] = cm_dt + neighbours * p.gAxial;
        d[c] = cm_dt * s.V[c] - ionic_current(s, c, p);
    }
    // Soma (compartment 0) also carries the synaptic term, implicit in V_new[0]:
    //   ... + g_syn*(eSyn - V_new[0])  ->  +g_syn on diagonal, +g_syn*eSyn on RHS.
    b[0] += s.g_syn;
    d[0] += s.g_syn * p.eSyn;

    thomas_solve(a, b, csup, d, cc, N);   // solution overwrites d[]
    for (int c = 0; c < N; ++c) s.V[c] = d[c];

    // (4) Spike = soma crossed threshold upward this step (was below, now >=).
    return (v_soma_old < p.vThresh && s.V[0] >= p.vThresh) ? 1 : 0;
}
