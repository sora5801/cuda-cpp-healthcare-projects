// ===========================================================================
// src/reference_cpu.cpp  --  Loader + serial split-step monodomain simulation
// ---------------------------------------------------------------------------
// Project 6.14 : Multi-Scale Physiological Modeling
// Compiled by the host compiler only. All per-node physics lives in
// multiscale.h (shared __host__ __device__), so this serial reference and the
// GPU kernel run byte-for-byte identical arithmetic per node.
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>
#include <stdexcept>

// ---------------------------------------------------------------------------
// load_cable : parse the whitespace-separated sample file.
//   Layout (see data/README.md):  n dx dt steps stim_nodes a eps b D
//   Every field is validated; a bad/short file throws so the demo fails loudly
//   rather than silently simulating garbage.
// ---------------------------------------------------------------------------
CableConfig load_cable(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open cable file: " + path);

    CableConfig c;
    if (!(in >> c.n >> c.dx >> c.dt >> c.steps >> c.stim_nodes
             >> c.p.a >> c.p.eps >> c.p.b >> c.p.D))
        throw std::runtime_error(
            "bad parameters (expected 'n dx dt steps stim_nodes a eps b D') in " + path);

    // Physical/structural sanity: a cable needs at least 3 nodes for a Laplacian,
    // positive spacing/step, at least one step, and a stimulus that fits.
    if (c.n < 3 || c.dx <= 0.0 || c.dt <= 0.0 || c.steps < 1 ||
        c.stim_nodes < 1 || c.stim_nodes > c.n)
        throw std::runtime_error("invalid cable parameters in " + path);

    return c;
}

// ---------------------------------------------------------------------------
// summarize_activation : turn a per-node activation map into the two headline
//   numbers. n_activated is simply the count of nodes that ever crossed
//   threshold. The conduction velocity is estimated over the "clean" interior
//   span of activated nodes (we skip the stimulated block and the very last
//   node) as distance / (delta activation time):
//        CV = (x_hi - x_lo) / (t_activation[hi] - t_activation[lo])
//   This is exactly how experimentalists measure conduction velocity from an
//   activation map. If fewer than two interior nodes activated, CV is 0.
// ---------------------------------------------------------------------------
void summarize_activation(const CableConfig& c,
                          const std::vector<double>& activation_time,
                          int& n_activated, double& conduction_velocity) {
    n_activated = 0;
    for (int i = 0; i < c.n; ++i)
        if (activation_time[i] >= 0.0) ++n_activated;

    conduction_velocity = 0.0;

    // Measurement window: start just past the stimulated block (so we measure
    // PROPAGATION, not the instantaneous stimulus), end one node before the
    // far boundary (to avoid the reflecting-boundary end effect).
    const int lo = c.stim_nodes;        // first non-stimulated node
    const int hi = c.n - 2;             // last interior node
    if (lo >= hi) return;               // cable too short to measure
    if (activation_time[lo] < 0.0 || activation_time[hi] < 0.0) return;

    const double dt_travel = activation_time[hi] - activation_time[lo];
    if (dt_travel <= 0.0) return;       // no forward propagation measured
    const double dist = (hi - lo) * c.dx;
    conduction_velocity = dist / dt_travel;
}

// ---------------------------------------------------------------------------
// simulate_cpu : the serial baseline. One global step = OPERATOR SPLITTING:
//     (A) REACTION half: advance every node's cell ODE (FHN via RK4) using its
//         local (v,w) only -- the fine, sub-grid scale. Fully independent per
//         node (this is what the GPU parallelizes over threads).
//     (B) DIFFUSION half: apply the tissue coupling by an explicit forward-Euler
//         step of  dv = dt * D * laplacian(v)  using a SNAPSHOT of v (so every
//         node sees the same "old" field -- a Jacobi-style update; using the
//         already-updated values would make the result order-dependent and
//         would NOT match the GPU, which is inherently parallel).
//   The recovery variable w is untouched by diffusion (only voltage spreads).
//
//   We record each node's activation time the first time v crosses 0.5.
// ---------------------------------------------------------------------------
void simulate_cpu(const CableConfig& c, CableResult& out) {
    const int n = c.n;

    // State fields. `v` is the excitation (voltage-like) variable that both
    // reacts locally and diffuses spatially; `w` is the slow recovery variable.
    std::vector<double> v(n, 0.0);
    std::vector<double> w(n, 0.0);
    // Scratch snapshot of v for the Jacobi diffusion sweep (read old, write new).
    std::vector<double> v_old(n, 0.0);

    // Activation map: -1 = "not yet activated".
    out.activation_time.assign(n, -1.0);

    // Initial condition: stimulate the first `stim_nodes` nodes to v=1 so an
    // action potential is born at the left end and travels rightward. Those
    // nodes are "active" at t=0.
    for (int i = 0; i < c.stim_nodes; ++i) { v[i] = 1.0; out.activation_time[i] = 0.0; }

    // March the coupled system forward in time.
    for (int s = 1; s <= c.steps; ++s) {
        const double t = s * c.dt;   // time at the END of this global step

        // (A) REACTION sub-step: independent cell ODE at every node.
        for (int i = 0; i < n; ++i)
            react_rk4_step(v[i], w[i], c.p, c.dt);

        // (B) DIFFUSION sub-step: Jacobi/forward-Euler on a snapshot of v.
        for (int i = 0; i < n; ++i) v_old[i] = v[i];
        for (int i = 0; i < n; ++i) {
            const double left  = mirror_left(v_old.data(), n, i);
            const double right = mirror_right(v_old.data(), n, i);
            const double lap   = diffusion_laplacian(left, v_old[i], right, c.dx);
            v[i] = v_old[i] + c.dt * c.p.D * lap;
        }

        // Record first-crossing (activation) times for any newly-excited node.
        for (int i = 0; i < n; ++i)
            if (out.activation_time[i] < 0.0 && v[i] >= 0.5)
                out.activation_time[i] = t;
    }

    // Snapshot the final field and derive the summary metrics.
    out.v_final = v;
    out.w_final = w;
    summarize_activation(c, out.activation_time, out.n_activated, out.conduction_velocity);
}
