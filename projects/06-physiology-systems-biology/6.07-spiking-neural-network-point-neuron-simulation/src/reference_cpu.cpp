// ===========================================================================
// src/reference_cpu.cpp  --  Serial CPU reference for the LIF network
// ---------------------------------------------------------------------------
// Project 6.7 : Spiking Neural Network (Point-Neuron) Simulation
//
// WHAT THIS FILE IS
//   The trusted, readable baseline. It simulates the SAME network the GPU does,
//   one neuron at a time, one timestep at a time, using the shared per-neuron
//   physics in lif.h and the SAME fixed-point synaptic accumulation. Because both
//   sides call lif_step() and both sum synaptic input as int64 fixed-point, the
//   CPU and GPU spike counts agree EXACTLY (verification tolerance == 0).
//
//   The serial loop here is also the clearest statement of the algorithm; read it
//   before kernels.cu, which is just this loop parallelised.
//
// READ THIS AFTER: lif.h, reference_cpu.h. READ NEXT: kernels.cu.
// ===========================================================================
#include "reference_cpu.h"

#include <algorithm>   // std::fill, std::swap
#include <fstream>     // std::ifstream
#include <stdexcept>   // std::runtime_error
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// load_network: parse the 4-line text config (format documented in the header
//   and data/README.md). We read token by token so extra whitespace/newlines are
//   harmless, and validate the essential invariants so a bad sample fails loudly
//   instead of silently simulating nonsense.
// ---------------------------------------------------------------------------
NetworkConfig load_network(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open network file: " + path);

    NetworkConfig c;
    // A tiny helper: read the next whitespace-separated double or throw.
    auto need = [&](const char* what) -> double {
        double x;
        if (!(in >> x)) throw std::runtime_error(std::string("malformed config (expected ") + what + ")");
        return x;
    };

    c.n_exc      = (int)need("n_exc");
    c.n_inh      = (int)need("n_inh");
    c.out_degree = (int)need("out_degree");

    c.w_exc     = need("w_exc");
    c.w_inh     = need("w_inh");
    c.ext_kick  = need("ext_kick");
    c.ext_every = (int)need("ext_every");

    c.steps = (int)need("steps");
    c.seed  = (unsigned long long)need("seed");

    c.np.v_rest        = need("v_rest");
    c.np.v_reset       = need("v_reset");
    c.np.v_thresh      = need("v_thresh");
    c.np.tau_m         = need("tau_m");
    c.np.tau_syn       = need("tau_syn");
    c.np.r_m           = need("r_m");
    c.np.refractory_ms = need("refractory_ms");
    c.np.dt            = need("dt");

    // Validate: positive sizes, sane counts, a usable modulus for external drive.
    if (net_n(c) <= 0)           throw std::runtime_error("network must have >=1 neuron");
    if (c.out_degree < 0)        throw std::runtime_error("out_degree must be >= 0");
    if (c.out_degree > net_n(c)) throw std::runtime_error("out_degree cannot exceed neuron count");
    if (c.steps <= 0)            throw std::runtime_error("steps must be > 0");
    if (c.np.dt <= 0.0)          throw std::runtime_error("dt must be > 0");
    if (c.ext_every <= 0)        c.ext_every = 1;   // 0/neg -> drive every neuron every step
    return c;
}

// NOTE: the per-neuron initial condition (init_state) and the deterministic
// background drive (external_drive_fixed) are shared with the GPU and therefore
// live in lif.h as LIF_HD functions -- so the CPU and GPU start from identical
// states and see identical external input. We simply call them below.

// ---------------------------------------------------------------------------
// simulate_cpu: the whole serial simulation. THE ALGORITHM, spelled out.
//
//   State we carry between steps:
//     state[i]           : each neuron's (v, g, refractory) -- see lif.h
//     spiked_prev[i]     : did neuron i spike on the PREVIOUS step? (0/1)
//   Per-step scratch:
//     input_fixed[i]     : int64 fixed-point synaptic input arriving at neuron i
//                          THIS step (external drive + all incoming spikes).
//
//   Why a one-step synaptic DELAY? Spikes emitted on step t are delivered on step
//   t+1. This (a) matches the minimal conduction delay of one dt, and (b) makes
//   the update order irrelevant: we first read last step's spikes to build this
//   step's input, THEN update -- there is no read-after-write hazard, so the GPU
//   (which updates all neurons in parallel) gets the identical answer.
//
//   Per step t:
//     1. Zero input_fixed[].
//     2. External drive: add each neuron's deterministic background kick.
//     3. Deliver spikes: for every neuron s that spiked last step, add its weight
//        (fixed-point) into each of its out_degree targets.  <-- the GPU atomic step
//     4. Update: for every neuron, convert its input_fixed to a double and call
//        lif_step(); record whether it spiked (into spiked_now).
//     5. Tally: bump per-neuron and per-step spike counts; roll spiked_now -> prev.
// ---------------------------------------------------------------------------
void simulate_cpu(const NetworkConfig& c, SimResult& out) {
    const int n = net_n(c);

    std::vector<NeuronState> state(n);
    for (int i = 0; i < n; ++i) state[i] = init_state(c, i);

    std::vector<unsigned char> spiked_prev(n, 0), spiked_now(n, 0);
    std::vector<long long>     input_fixed(n, 0);

    out.total_spikes = 0;
    out.spikes_per_neuron.assign(n, 0);
    out.spikes_per_step.assign(c.steps, 0);
    out.final_v.assign(n, 0.0);

    for (int t = 0; t < c.steps; ++t) {
        // (1) zero this step's synaptic accumulator.
        std::fill(input_fixed.begin(), input_fixed.end(), 0LL);

        // (2) external background drive (deterministic).
        for (int i = 0; i < n; ++i)
            input_fixed[i] += external_drive_fixed(c, i, t);

        // (3) deliver last step's spikes along the sparse graph. This is exactly
        //     the "scatter with atomicAdd" the GPU does -- here it is a plain
        //     serial += because there is only one thread of execution.
        for (int s = 0; s < n; ++s) {
            if (!spiked_prev[s]) continue;
            const long long w_fixed = syn_to_fixed(src_weight(c, s));
            for (int k = 0; k < c.out_degree; ++k) {
                const int tgt = synapse_target(c, s, k);
                input_fixed[tgt] += w_fixed;      // integer add: order-independent
            }
        }

        // (4) update every neuron with the shared physics; record spikes.
        int step_spikes = 0;
        for (int i = 0; i < n; ++i) {
            const double drive = syn_from_fixed(input_fixed[i]);
            const int fired = lif_step(c.np, state[i], drive);
            spiked_now[i] = (unsigned char)fired;
            if (fired) {
                ++out.spikes_per_neuron[i];
                ++step_spikes;
            }
        }

        // (5) tally + roll the spike buffer forward (this step's spikes become
        //     next step's inputs -- the one-dt synaptic delay).
        out.spikes_per_step[t] = step_spikes;
        out.total_spikes += step_spikes;
        spiked_prev.swap(spiked_now);
    }

    for (int i = 0; i < n; ++i) out.final_v[i] = state[i].v;

    // Derived, human-facing metric: population mean firing rate over the run.
    //   mean_rate = total_spikes / (n * duration_seconds).
    const double duration_s = c.steps * c.np.dt * 1.0e-3;   // ms -> s
    out.mean_rate_hz = (n > 0 && duration_s > 0.0)
                     ? (double)out.total_spikes / ((double)n * duration_s)
                     : 0.0;
}
