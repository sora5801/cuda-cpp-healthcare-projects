// ===========================================================================
// src/lif.h  --  The ONE TRUE per-neuron physics, shared by CPU and GPU
// ---------------------------------------------------------------------------
// Project 6.7 : Spiking Neural Network (Point-Neuron) Simulation
//
// WHY THIS HEADER EXISTS (PATTERNS.md sec.2 -- the __host__ __device__ core)
//   The reference CPU code (reference_cpu.cpp, compiled by cl.exe) and the GPU
//   kernels (kernels.cu, compiled by nvcc) must produce BYTE-IDENTICAL neuron
//   trajectories, otherwise "GPU matches CPU" verification is meaningless. The
//   only way to guarantee that is to write the per-neuron math EXACTLY ONCE, in
//   inline functions that both compilers include. That is this file.
//
//   Rules for this header (so the plain host compiler can also read it):
//     * NO __global__ kernels, NO CUDA-only types here -- only plain C++ + the
//       LIF_HD decorator, which expands to __host__ __device__ under nvcc and to
//       nothing under cl.exe/g++.
//     * All math is double precision and uses the SAME operations on both sides
//       (no fast-math, no transcendental shortcuts) so the results agree exactly.
//
// THE MODEL (see ../THEORY.md "The science" and "The math")
//   Each neuron is a Leaky Integrate-and-Fire (LIF) point neuron: a single
//   membrane-potential variable V that leaks toward a resting value and jumps up
//   when synaptic input arrives. When V crosses a threshold it emits a SPIKE,
//   V is reset, and the neuron is briefly refractory (deaf to input). Synaptic
//   input is modelled as an exponentially-decaying current g (a "g-synapse").
//
//   This is the workhorse of large-scale brain simulation: it throws away the
//   neuron's dendritic geometry (that lives in projects like 6.x cable models)
//   and keeps just enough to reproduce network-level spiking dynamics -- which is
//   exactly what lets tools like GeNN/NEST simulate millions of neurons.
//
// READ THIS AFTER: nothing (start here). READ NEXT: reference_cpu.h, kernels.cuh.
// ===========================================================================
#pragma once

#include <cstdint>   // uint32_t, uint64_t  (fixed-point + hashing)

// ---------------------------------------------------------------------------
// LIF_HD: the portability shim. Under nvcc (__CUDACC__ defined) every function
// below is compiled for BOTH the host and the device; under a plain C++ compiler
// the decorators vanish and we get ordinary host functions. This is the idiom
// that guarantees CPU/GPU parity (PATTERNS.md sec.2).
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define LIF_HD __host__ __device__
#else
#define LIF_HD
#endif

// ---------------------------------------------------------------------------
// NeuronParams: the fixed biophysical constants of ONE population of LIF neurons.
//   All voltages are in millivolts (mV), all times in milliseconds (ms). We keep
//   them in one small struct so we can pass "the model" by value into kernels
//   (it lands in constant/param memory -- cheap and read-only).
// ---------------------------------------------------------------------------
struct NeuronParams {
    double v_rest = -65.0;    // resting / leak-reversal potential  [mV]
    double v_reset = -65.0;   // potential the neuron is reset to after a spike [mV]
    double v_thresh = -50.0;  // spike threshold: V >= this -> emit a spike [mV]
    double tau_m = 20.0;      // membrane time constant (how fast V leaks) [ms]
    double tau_syn = 5.0;     // synaptic-current decay time constant [ms]
    double r_m = 80.0;        // membrane resistance times unit -> converts g to mV drive [Mohm-ish, tuned]
    double refractory_ms = 2.0; // dead time after a spike during which V is clamped [ms]
    double dt = 0.1;          // integration timestep [ms]
};

// ---------------------------------------------------------------------------
// NeuronState: the TIME-VARYING state of ONE neuron. Small on purpose -- this is
// the whole point of a "point neuron": a handful of scalars per cell, so a
// million neurons fit comfortably in GPU memory.
//   v            : membrane potential [mV]
//   g            : synaptic conductance / input current variable [arbitrary drive units]
//   refrac_steps : integer countdown of remaining refractory timesteps (0 = active)
// We store the refractory clock as an INTEGER count of steps (not a float time)
// so the "is this neuron refractory?" test is exact and identical on CPU & GPU.
// ---------------------------------------------------------------------------
struct NeuronState {
    double v = -65.0;
    double g = 0.0;
    int    refrac_steps = 0;
};

// ---------------------------------------------------------------------------
// FIXED-POINT SYNAPTIC ACCUMULATION (the determinism trick, PATTERNS.md sec.3)
//   Every timestep, all neurons that spiked in the PREVIOUS step deposit their
//   synaptic weight into each of their postsynaptic targets. On the GPU thousands
//   of source threads atomicAdd into the same target neuron -- and floating-point
//   atomicAdd is NON-associative, so the order (which is nondeterministic across
//   warps) would change the sum in the last bits. That would make the GPU result
//   drift from the CPU and from run to run.
//
//   The fix: accumulate synaptic input as SCALED INTEGERS. Integer addition
//   commutes, so any order gives the same total, exactly matching the CPU's
//   serial sum. We convert weight -> int64 by multiplying by SYN_FIXED_SCALE,
//   sum in int64_t, then convert back to double once per neuron per step.
//
//   SYN_FIXED_SCALE = 2^20 (~1.05e6) gives ~6 decimal digits of resolution on the
//   weights, far finer than any biological weight precision, while a synapse count
//   of ~1e4 * a weight of ~1.0 stays well inside int64_t's ~9.2e18 range.
// ---------------------------------------------------------------------------
// constexpr (not just const) so nvcc treats it as a compile-time constant that is
// usable inside __device__ code, not a host-only global. 2^20 ~= 1.05e6.
constexpr double SYN_FIXED_SCALE = 1048576.0;   // 2^20

// Convert a real synaptic weight to its fixed-point integer representation.
LIF_HD inline long long syn_to_fixed(double w) {
    // Round-to-nearest (add 0.5 with the sign) so + and - weights are symmetric
    // and CPU/GPU agree bit-for-bit on the integer value of each weight.
    return (long long)(w * SYN_FIXED_SCALE + (w >= 0.0 ? 0.5 : -0.5));
}

// Convert an accumulated fixed-point sum back to a real current increment.
LIF_HD inline double syn_from_fixed(long long q) {
    return (double)q / SYN_FIXED_SCALE;
}

// ---------------------------------------------------------------------------
// lif_step: advance ONE neuron by ONE timestep. THE single source of truth.
//   Inputs:
//     p          : the population's fixed parameters (dt, tau_m, threshold, ...)
//     s          : this neuron's mutable state (updated in place)
//     input_drive: the synaptic current delivered to this neuron THIS step, as a
//                  real number already summed from the fixed-point accumulator
//                  (units of conductance/drive; added onto g).
//   Returns: 1 if the neuron SPIKED this step, else 0.
//
//   Numerical method: EXPONENTIAL EULER (a.k.a. exact integration for the linear
//   leak). The LIF sub-threshold equation
//        tau_m dV/dt = (v_rest - V) + r_m * g
//   is linear in V over a step where g is held constant, so its exact update is
//        V(t+dt) = v_inf + (V(t) - v_inf) * exp(-dt/tau_m),
//   with v_inf = v_rest + r_m * g the steady state. This is more stable than
//   forward Euler (never overshoots for any dt) and, crucially, is a fixed set of
//   arithmetic ops we can reproduce identically on host and device. The synaptic
//   conductance g decays multiplicatively by exp(-dt/tau_syn) each step.
//
//   Order of operations MATTERS for CPU/GPU parity, so it is fixed here:
//     1. inject this step's synaptic input onto g
//     2. if refractory: clamp V to reset, tick down the refractory clock, no spike
//     3. else: exponential-Euler update of V using the *pre-decay* g
//     4. decay g for next step
//     5. threshold test -> maybe spike (reset V, arm refractory)
// ---------------------------------------------------------------------------
LIF_HD inline int lif_step(const NeuronParams& p, NeuronState& s, double input_drive) {
    // (1) Deliver this step's synaptic input. Because input_drive came from an
    //     integer fixed-point sum, it is identical on CPU and GPU.
    s.g += input_drive;

    // Precompute the two decay factors. These are pure functions of the (shared)
    // parameters, so host and device compute the same bits. exp() is called on
    // both sides identically (no fast-math on this project -- see the .vcxproj).
    const double leak_decay = exp(-p.dt / p.tau_m);    // membrane relaxation factor
    const double syn_decay  = exp(-p.dt / p.tau_syn);  // synaptic decay factor

    int spiked = 0;

    if (s.refrac_steps > 0) {
        // (2) Refractory period: the neuron cannot integrate input. We hold V at
        //     the reset potential and simply count down. Synaptic g still decays
        //     (channels keep closing) so no input is lost to the model's logic.
        s.v = p.v_reset;
        s.refrac_steps -= 1;
    } else {
        // (3) Sub-threshold exponential-Euler membrane update. v_inf is the value
        //     V would relax to if g were held fixed forever.
        const double v_inf = p.v_rest + p.r_m * s.g;
        s.v = v_inf + (s.v - v_inf) * leak_decay;

        // (5) Threshold crossing -> spike. We test AFTER the update so a strong
        //     input in this step can trigger a spike in this step (standard LIF).
        if (s.v >= p.v_thresh) {
            s.v = p.v_reset;                                  // reset the membrane
            // Arm the refractory clock. We round refractory_ms/dt to the nearest
            // integer number of steps; both sides do the identical rounding.
            s.refrac_steps = (int)(p.refractory_ms / p.dt + 0.5);
            spiked = 1;
        }
    }

    // (4) Decay the synaptic conductance for the NEXT step. Done last so the spike
    //     test above saw this step's full g.
    s.g *= syn_decay;

    return spiked;
}

// ---------------------------------------------------------------------------
// DETERMINISTIC SPARSE CONNECTIVITY (shared by CPU and GPU)
//   A Brunel random balanced network: each neuron makes a FIXED number of
//   outgoing synapses (out_degree) to random targets. To keep the CPU and GPU
//   building the *same* graph without shipping a big matrix, we DERIVE each
//   synapse's target from a cheap, stateless hash of (source, synapse-index).
//   Both sides call the same hash, so both see the same wiring -- no random-state
//   divergence, no stored connectivity matrix (memory: O(0) beyond the params).
//
//   This is SplitMix64, a well-known fast integer mixer with good avalanche.
//   It is deterministic and identical on host and device (pure uint64 math).
// ---------------------------------------------------------------------------
LIF_HD inline uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;             // golden-ratio increment
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

// ---------------------------------------------------------------------------
// NetworkConfig: the whole simulated network in a handful of scalars.
//   n_exc / n_inh : counts of excitatory and inhibitory neurons (n = sum).
//   out_degree    : synapses each neuron sends (sparse: << n).
//   w_exc / w_inh : synaptic weights (inh is negative -> hyperpolarizing).
//   ext_rate_hz   : rate of an external Poisson-like drive every neuron receives
//                   (kept DETERMINISTIC here: a fixed per-step kick pattern, see
//                   reference_cpu.cpp) so the demo output is reproducible.
//   steps         : number of timesteps to simulate (duration = steps*dt ms).
//   seed          : hashes the connectivity + the initial-state jitter.
// The layout convention: neuron ids [0 .. n_exc) are excitatory, [n_exc .. n)
// are inhibitory. A neuron's OUTGOING weight is set by which group IT is in.
// ---------------------------------------------------------------------------
struct NetworkConfig {
    int n_exc = 0;
    int n_inh = 0;
    int out_degree = 0;
    double w_exc = 0.0;
    double w_inh = 0.0;
    double ext_kick = 0.0;   // fixed external drive added to a subset each step
    int    ext_every = 0;    // ...applied to neurons whose id % ext_every == step % ext_every
    int    steps = 0;
    uint64_t seed = 0;
    NeuronParams np;         // the biophysics (dt lives here)
};

// Total neuron count.
LIF_HD inline int net_n(const NetworkConfig& c) { return c.n_exc + c.n_inh; }

// Is neuron `id` excitatory? (Determines the sign/size of its outgoing weight.)
LIF_HD inline bool is_excitatory(const NetworkConfig& c, int id) { return id < c.n_exc; }

// The outgoing synaptic weight of source neuron `src` (same for all its synapses).
LIF_HD inline double src_weight(const NetworkConfig& c, int src) {
    return is_excitatory(c, src) ? c.w_exc : c.w_inh;
}

// Target of the k-th outgoing synapse of source neuron `src`.
//   We hash (seed, src, k) -> a target in [0, n). Self-connections are allowed but
//   harmless here (a neuron nudging itself); a production wiring would reject them.
//   Because host and device call this identical function, both build the SAME graph.
LIF_HD inline int synapse_target(const NetworkConfig& c, int src, int k) {
    uint64_t h = splitmix64(c.seed ^ (0xD1B54A32D192ED03ULL * (uint64_t)src + (uint64_t)k * 0x2545F4914F6CDD1DULL));
    return (int)(h % (uint64_t)net_n(c));
}

// ---------------------------------------------------------------------------
// init_state: deterministic initial condition for neuron `id` (shared CPU/GPU).
//   We jitter each neuron's starting membrane potential between v_reset and
//   v_thresh from a hash of (seed, id) so the network does not start in perfect
//   lockstep (which would create an artificial synchronous first volley). Because
//   this is LIF_HD, the CPU reference and the GPU init kernel produce identical
//   starting states -- a prerequisite for the trajectories matching exactly.
// ---------------------------------------------------------------------------
LIF_HD inline NeuronState init_state(const NetworkConfig& c, int id) {
    NeuronState s;
    // 64-bit hash -> a 53-bit fraction in [0,1) -> a voltage in [v_reset, v_thresh).
    uint64_t h = splitmix64(c.seed ^ (0x1234567ULL + 0x9E3779B1ULL * (uint64_t)id));
    double frac = (double)(h >> 11) * (1.0 / 9007199254740992.0);   // 2^-53 * (top 53 bits)
    s.v = c.np.v_reset + frac * (c.np.v_thresh - c.np.v_reset);
    s.g = 0.0;
    s.refrac_steps = 0;
    return s;
}

// ---------------------------------------------------------------------------
// external_drive_fixed: DETERMINISTIC stand-in for background Poisson input.
//   A real SNN drives each neuron with an independent Poisson spike train (cuRAND
//   on the GPU). Poisson RNG is fine but makes the demo output depend on the RNG
//   implementation, which we avoid for a byte-reproducible demo. Instead we apply
//   a fixed, phase-rotating kick: on step `t`, every neuron whose id has the same
//   residue as t modulo ext_every gets a kick of ext_kick. This is identical on
//   CPU and GPU and identical every run. (THEORY: swap in true cuRAND Poisson as
//   an exercise.) Returned as a fixed-point integer, to be summed with synapses.
// ---------------------------------------------------------------------------
LIF_HD inline long long external_drive_fixed(const NetworkConfig& c, int id, int t) {
    if (c.ext_kick == 0.0) return 0;
    if ((id % c.ext_every) == (t % c.ext_every)) return syn_to_fixed(c.ext_kick);
    return 0;
}
