// ===========================================================================
// src/reference_cpu.h  --  Data model + CPU reference API for the coronary solve
// ---------------------------------------------------------------------------
// Project 6.20 : Coronary Autoregulation & Microvascular Perfusion
//
// ROLE
//   Declares (a) the plain-old-data structures that hold a vascular network and
//   its solution, and (b) the CPU reference routines that main.cu calls to
//   produce the ground-truth answer the GPU is checked against.
//
//   The CPU reference is the TEACHING BASELINE: it does exactly what the GPU
//   does (assemble the graph-Laplacian, solve L p = b with Conjugate Gradient,
//   run the autoregulation outer loop) but in ordinary serial C++ you can read
//   top-to-bottom. Because both sides call the SAME per-vessel physics in
//   coronary.h, agreement is expected to ~machine precision (see main.cu).
//
// READ THIS AFTER: coronary.h. READ BEFORE: reference_cpu.cpp, main.cu.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Network: a coronary microvascular graph in a solver-friendly layout.
//
//   NODES are junctions where segments meet. Some nodes are BOUNDARY nodes with
//   a prescribed (Dirichlet) pressure: the coronary INLET (aortic pressure) and
//   the venous/capillary OUTLETS (a low back-pressure, our lumped "structured-
//   tree Windkessel" terminal). Interior nodes obey flow conservation.
//
//   SEGMENTS are cylindrical vessels, each connecting node `a` to node `b`, with
//   a radius and length. Conductance is derived from these via coronary.h.
//
// We store segments as parallel arrays (structure-of-arrays) because that is
// the layout the GPU wants: thread s reads seg_a[s], seg_b[s], seg_r[s], ... as
// coalesced loads. The CPU reference uses the very same arrays.
// ---------------------------------------------------------------------------
struct Network {
    int n_nodes = 0;                 // number of junctions
    int n_segs  = 0;                 // number of vessel segments (graph edges)

    // Per-segment topology + geometry (length n_segs each).
    std::vector<int>    seg_a;       // endpoint node index A
    std::vector<int>    seg_b;       // endpoint node index B
    std::vector<double> seg_r;       // radius (um)   -- MUTATED by autoregulation
    std::vector<double> seg_len;     // length (um)
    std::vector<double> seg_target;  // metabolic target |flow| set-point (um^3/s)

    // Per-node boundary conditions (length n_nodes).
    //   is_fixed[i] != 0  => node i has a prescribed pressure fixed_p[i].
    std::vector<int>    is_fixed;    // 1 = Dirichlet boundary, 0 = interior
    std::vector<double> fixed_p;     // prescribed pressure (mmHg) where fixed

    double hct = 0.45;               // hematocrit fraction (feeds viscosity)

    // For the demo's FFR readout we tag one segment as the "stenosis" and record
    // the two nodes across which the pressure ratio (FFR) is measured.
    int    ffr_seg     = -1;         // segment index of the modeled stenosis
    int    ffr_prox    = -1;         // proximal (upstream) node
    int    ffr_dist    = -1;         // distal (downstream) node
    double aortic_p    = 0.0;        // inlet pressure (mmHg), for FFR normalization
};

// Solution: nodal pressures + derived per-segment flows for one solve.
struct Solution {
    std::vector<double> p;    // [n_nodes] pressures (mmHg), final state
    std::vector<double> q;    // [n_segs]  signed flows a->b (um^3/s), final state
    int    cg_iters = 0;      // Conjugate-Gradient iterations used (last solve)
    double cg_resid = 0.0;    // final CG residual norm (last solve)
    int    cg_iters_first = 0;// CG iterations of the FIRST (cold-start) solve --
                              //   the honest "how hard is the sparse solve" figure
                              //   (later solves warm-start from the previous p and
                              //    converge in ~0 iters). Reported to stderr.
    double perfusion_first = 0.0; // total inlet perfusion BEFORE autoregulation
                              //   (after the k=0 solve), so the demo can show the
                              //   regulated perfusion moving toward the set-point.
};

// ---------------------------------------------------------------------------
// load_network(path)
//   Parse the tiny whitespace-delimited sample (data/sample/*.txt) into a
//   Network. Format is documented in data/README.md and scripts/make_synthetic.py.
//   Throws std::runtime_error on a malformed file so the demo fails loudly.
// ---------------------------------------------------------------------------
Network load_network(const std::string& path);

// ---------------------------------------------------------------------------
// solve_cpu(net, n_autoreg, cg_tol, cg_max_iter, out)
//   The full CPU reference: run `n_autoreg` outer autoregulation steps; within
//   each, assemble the graph-Laplacian for the current radii and solve
//   L p = b with Conjugate Gradient, then update radii from the resulting flows.
//   Writes the final pressures/flows into `out`. This is the ground truth.
//
//   n_autoreg   : number of autoregulation outer iterations (>= 1; the last
//                 iteration does NOT change radii so p,q are self-consistent)
//   cg_tol      : CG relative-residual stopping tolerance
//   cg_max_iter : CG iteration cap
// ---------------------------------------------------------------------------
void solve_cpu(Network& net, int n_autoreg, double cg_tol, int cg_max_iter,
               Solution& out);

// ---------------------------------------------------------------------------
// compute_flows(net, p, q)
//   Given nodal pressures, compute each segment's signed flow Q = G*(p_a - p_b).
//   Shared by the autoregulation loop and the final report. O(n_segs).
// ---------------------------------------------------------------------------
void compute_flows(const Network& net, const std::vector<double>& p,
                   std::vector<double>& q);

// ---------------------------------------------------------------------------
// inlet_node(net) / inlet_perfusion(net, q)
//   Helpers for the perfusion read-out. The inlet is the fixed node with the
//   highest prescribed pressure (aortic side). Perfusion = net flow LEAVING that
//   node (summed over its incident segments, signed by orientation). Defined
//   inline in the header so main.cu (nvcc) and reference_cpu.cpp (host) share
//   one definition. O(n_segs).
// ---------------------------------------------------------------------------
inline int inlet_node(const Network& net) {
    int inlet = 0; double pmax = -1e300;
    for (int i = 0; i < net.n_nodes; ++i)
        if (net.is_fixed[i] && net.fixed_p[i] > pmax) { pmax = net.fixed_p[i]; inlet = i; }
    return inlet;
}
inline double inlet_perfusion(const Network& net, const std::vector<double>& q) {
    const int inlet = inlet_node(net);
    double perf = 0.0;
    for (int s = 0; s < net.n_segs; ++s) {
        if      (net.seg_a[s] == inlet) perf += q[s];   // segment oriented out of inlet
        else if (net.seg_b[s] == inlet) perf -= q[s];   // oriented into inlet -> outflow is -q
    }
    return perf;
}
