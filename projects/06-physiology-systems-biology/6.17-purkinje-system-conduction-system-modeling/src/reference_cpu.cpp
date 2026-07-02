// ===========================================================================
// src/reference_cpu.cpp  --  Loader + serial cable simulation + graph delays
// ---------------------------------------------------------------------------
// Project 6.17 : Purkinje System & Conduction System Modeling
//
// Compiled by the host compiler ONLY. The per-cable PDE/stepper lives in
// purkinje.h (shared host+device via PK_HD); this file just (a) parses the tiny
// tree file, (b) loops over cables calling the shared stepper -- the trusted
// serial baseline the GPU kernel is checked against -- and (c) walks the tree
// graph to turn per-cable delays into absolute activation times.
//
// READ THIS AFTER: reference_cpu.h, purkinje.h. Compare against kernels.cu.
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>
#include <stdexcept>

// ---------------------------------------------------------------------------
// load_tree  --  parse the tiny whitespace-delimited tree description.
//   Format (see data/README.md):
//     N dt_ms n_steps
//     <N lines>: n_nodes length_mm D stim_amp stim_dur_ms stim_width thresh parent delay_ms
//   The global dt_ms / n_steps are copied into every cable (they share the clock)
//   so the shared stepper needs only a CableParams. We validate aggressively so a
//   malformed sample fails at load time with a clear message, not mid-kernel.
// ---------------------------------------------------------------------------
PurkinjeTree load_tree(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open tree file: " + path);

    int    N = 0, n_steps = 0;
    double dt_ms = 0.0;
    if (!(in >> N >> dt_ms >> n_steps))
        throw std::runtime_error("bad header (expected 'N dt_ms n_steps') in " + path);
    if (N <= 0 || dt_ms <= 0.0 || n_steps <= 0)
        throw std::runtime_error("invalid header values in " + path);

    PurkinjeTree t;
    t.cables.reserve(N);
    for (int i = 0; i < N; ++i) {
        CableParams c;
        // Per-cable fields. `parent` and `delay_ms` describe the tree edge that
        // feeds this cable; the rest describe its 1-D cable simulation.
        if (!(in >> c.n_nodes >> c.length_mm >> c.D >> c.stim_amp
                 >> c.stim_dur_ms >> c.stim_width >> c.thresh
                 >> c.parent >> c.delay_ms))
            throw std::runtime_error("bad cable line " + std::to_string(i) + " in " + path);

        // Fill in the shared global clock and sanity-check bounds.
        c.dt_ms   = dt_ms;
        c.n_steps = n_steps;
        if (c.n_nodes < 2 || c.n_nodes > PK_MAX_NODES)
            throw std::runtime_error("cable " + std::to_string(i)
                + ": n_nodes out of range [2," + std::to_string(PK_MAX_NODES) + "]");
        if (c.length_mm <= 0.0 || c.D < 0.0)
            throw std::runtime_error("cable " + std::to_string(i) + ": bad length/D");
        // The tree must be topologically ordered (parent index < child index) so a
        // single forward pass computes activation times. Roots use parent == -1.
        if (c.parent >= i)
            throw std::runtime_error("cable " + std::to_string(i)
                + ": parent must be < index (tree not topologically ordered)");
        t.cables.push_back(c);
    }
    return t;
}

// ---------------------------------------------------------------------------
// simulate_cpu  --  the serial reference: run every cable's PDE independently.
//   Each cable gets its own scratch buffers on the stack; there is no coupling
//   during the PDE solve (coupling happens afterwards, in the graph-delay pass).
//   This is exactly what the GPU kernel does, one thread per cable, so the two
//   produce identical CableResults.
// ---------------------------------------------------------------------------
void simulate_cpu(const PurkinjeTree& t, std::vector<CableResult>& results) {
    const int N = tree_size(t);
    results.assign(N, CableResult{});

    // Per-cable scratch: two voltage buffers (ping-pong) + one recovery buffer.
    // Sized to the fixed maximum so the layout matches the GPU's local arrays.
    double Va[PK_MAX_NODES], Vb[PK_MAX_NODES], w[PK_MAX_NODES];

    for (int i = 0; i < N; ++i) {
        results[i] = pk_simulate_cable(t.cables[i], Va, Vb, w);
    }
}

// ---------------------------------------------------------------------------
// compute_activation_times  --  O(N) forward pass over the rooted tree.
//   local_delay[i] = (step_out - step_in) * dt  is the time the front takes to
//   traverse cable i (its length / CV). Because parents precede children in
//   index order, one left-to-right sweep resolves every absolute time.
//
//   A cable that BLOCKED (never activated its distal end) has an undefined out
//   time; we mark its distal activation as -1 so downstream analysis can see the
//   conduction failure rather than silently propagating a garbage time.
// ---------------------------------------------------------------------------
std::vector<double> compute_activation_times(const PurkinjeTree& t,
                                             const std::vector<CableResult>& res) {
    const int N = tree_size(t);
    std::vector<double> t_in(N, 0.0);    // absolute time the proximal end activates (ms)
    std::vector<double> t_out(N, -1.0);  // absolute time the distal end (PMJ) activates (ms)

    for (int i = 0; i < N; ++i) {
        const CableParams& c = t.cables[i];

        // When does this cable's proximal end become active?
        if (c.parent < 0) {
            // Root (His bundle): paced directly, offset by its junction delay.
            t_in[i] = c.delay_ms;
        } else {
            // Fed by the parent's distal end; if the parent blocked, so do we.
            if (t_out[c.parent] < 0.0) { t_out[i] = -1.0; continue; }
            t_in[i] = t_out[c.parent] + c.delay_ms;
        }

        // Local traversal time from the measured activation-step gap.
        if (res[i].captured && res[i].activate_step_out >= res[i].activate_step_in
            && res[i].activate_step_in >= 0) {
            const double local_ms =
                (double)(res[i].activate_step_out - res[i].activate_step_in) * c.dt_ms;
            t_out[i] = t_in[i] + local_ms;
        } else {
            t_out[i] = -1.0;   // conduction block within this cable
        }
    }
    return t_out;
}
