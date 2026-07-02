// ===========================================================================
// src/reference_cpu.cpp  --  Serial ground-truth: assemble + CG-solve + regulate
// ---------------------------------------------------------------------------
// Project 6.20 : Coronary Autoregulation & Microvascular Perfusion
//
// WHAT THIS FILE COMPUTES
//   The complete coronary perfusion problem on the CPU, in readable serial C++:
//
//     for k in 0..n_autoreg-1:
//         G_s   = segment_conductance(r_s, L_s, hct)      # coronary.h (r^4 law)
//         L,b   = assemble graph-Laplacian with Dirichlet BCs
//         p     = ConjugateGradient(L, b)                 # sparse SPD solve
//         q_s   = G_s * (p_a - p_b)                        # Poiseuille flows
//         if k < last:  r_s = autoregulate_radius(...)     # feedback (coronary.h)
//
//   The final (p, q) is the ground truth the GPU must reproduce. Every formula
//   that could drift between CPU and GPU lives in coronary.h and is called from
//   BOTH, so the only differences are floating-point summation order (which we
//   bound with a documented tolerance in main.cu).
//
// WHY GRAPH-LAPLACIAN? (the math the learner should carry away)
//   Flow conservation at interior node i:  sum_over_segments_at_i G_s (p_i - p_j) = 0.
//   Collecting terms gives, for interior rows,
//       (sum_s G_s) p_i  -  sum_j G_ij p_j  = 0,
//   i.e. the weighted graph Laplacian L with L_ii = sum of incident conductances
//   and L_ij = -G_ij. L is symmetric; with at least one Dirichlet (fixed-pressure)
//   node it is positive-definite -> Conjugate Gradient is the textbook solver.
//   Boundary nodes are handled by pinning the row to the identity (p_i = fixed).
//
// READ THIS AFTER: coronary.h, reference_cpu.h. Mirrors kernels.cu on the GPU.
// ===========================================================================
#include "reference_cpu.h"
#include "coronary.h"

#include <cmath>       // std::sqrt, std::fabs
#include <fstream>     // std::ifstream
#include <sstream>     // std::istringstream
#include <stdexcept>   // std::runtime_error
#include <string>

// ===========================================================================
// SECTION 1 -- Loading the network from the tiny text sample
// ---------------------------------------------------------------------------
// The sample format (see data/README.md) is line-oriented and human-editable:
//   line 1:  n_nodes n_segs hct aortic_p
//   next n_nodes lines:   is_fixed  fixed_p          (one per node, node 0..N-1)
//   next n_segs  lines:   a b radius length target   (one per segment)
//   last line:            ffr_seg ffr_prox ffr_dist  (indices for the FFR readout)
// Lines beginning with '#' are comments and are skipped. Keeping the parser
// hand-rolled (rather than a library) keeps the project self-contained.
// ===========================================================================
namespace {

// Read the next non-comment, non-blank line into `line`. Returns false at EOF.
bool next_line(std::ifstream& in, std::string& line) {
    while (std::getline(in, line)) {
        // Trim a leading UTF-8 BOM if present on the very first line (Windows
        // editors love to add it) so the first integer parses cleanly.
        if (line.size() >= 3 &&
            static_cast<unsigned char>(line[0]) == 0xEF &&
            static_cast<unsigned char>(line[1]) == 0xBB &&
            static_cast<unsigned char>(line[2]) == 0xBF) {
            line.erase(0, 3);
        }
        // Skip blank lines and full-line comments.
        std::size_t first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) continue;   // blank
        if (line[first] == '#') continue;           // comment
        return true;
    }
    return false;
}

}  // namespace

Network load_network(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open network file: " + path);

    Network net;
    std::string line;

    // ---- header ----
    if (!next_line(in, line)) throw std::runtime_error("empty network file: " + path);
    {
        std::istringstream ss(line);
        if (!(ss >> net.n_nodes >> net.n_segs >> net.hct >> net.aortic_p))
            throw std::runtime_error("bad header line in: " + path);
    }
    if (net.n_nodes <= 0 || net.n_segs <= 0)
        throw std::runtime_error("non-positive node/segment count in: " + path);

    // ---- per-node boundary conditions ----
    net.is_fixed.resize(net.n_nodes);
    net.fixed_p.resize(net.n_nodes, 0.0);
    for (int i = 0; i < net.n_nodes; ++i) {
        if (!next_line(in, line)) throw std::runtime_error("missing node line in: " + path);
        std::istringstream ss(line);
        int fixed; double p;
        if (!(ss >> fixed >> p)) throw std::runtime_error("bad node line in: " + path);
        net.is_fixed[i] = fixed;
        net.fixed_p[i]  = p;
    }

    // ---- per-segment geometry ----
    net.seg_a.resize(net.n_segs);
    net.seg_b.resize(net.n_segs);
    net.seg_r.resize(net.n_segs);
    net.seg_len.resize(net.n_segs);
    net.seg_target.resize(net.n_segs);
    for (int s = 0; s < net.n_segs; ++s) {
        if (!next_line(in, line)) throw std::runtime_error("missing segment line in: " + path);
        std::istringstream ss(line);
        int a, b; double r, L, tgt;
        if (!(ss >> a >> b >> r >> L >> tgt))
            throw std::runtime_error("bad segment line in: " + path);
        if (a < 0 || a >= net.n_nodes || b < 0 || b >= net.n_nodes)
            throw std::runtime_error("segment endpoint out of range in: " + path);
        net.seg_a[s] = a; net.seg_b[s] = b;
        net.seg_r[s] = r; net.seg_len[s] = L; net.seg_target[s] = tgt;
    }

    // ---- FFR readout indices ----
    if (!next_line(in, line)) throw std::runtime_error("missing FFR line in: " + path);
    {
        std::istringstream ss(line);
        if (!(ss >> net.ffr_seg >> net.ffr_prox >> net.ffr_dist))
            throw std::runtime_error("bad FFR line in: " + path);
    }
    return net;
}

// ===========================================================================
// SECTION 2 -- Compute per-segment flows from nodal pressures
// ---------------------------------------------------------------------------
// Q_s = G_s * (p[a] - p[b]). Positive means flow from a to b. Uses the SAME
// conductance formula (coronary.h) the GPU uses.
// ===========================================================================
void compute_flows(const Network& net, const std::vector<double>& p,
                   std::vector<double>& q) {
    q.assign(net.n_segs, 0.0);
    for (int s = 0; s < net.n_segs; ++s) {
        const double G = coronary::segment_conductance(net.seg_r[s], net.seg_len[s], net.hct);
        q[s] = G * (p[net.seg_a[s]] - p[net.seg_b[s]]);
    }
}

// ===========================================================================
// SECTION 3 -- Assemble the graph-Laplacian, then Conjugate-Gradient
// ---------------------------------------------------------------------------
// We apply L "matrix-free" via the SAME edge list the GPU uses -- this is the
// honest sparse mat-vec (SpMV) and keeps the two implementations structurally
// identical. THEORY §GPU-mapping explains why the GPU instead builds an explicit
// CSR (one thread per node needs O(1) access to a node's incident edges).
// ===========================================================================
namespace {

// spmv(net, G, x, y): y = L x, where L is the boundary-eliminated graph-Laplacian.
//   To keep L SYMMETRIC POSITIVE-DEFINITE (the precondition CG needs), we
//   ELIMINATE the fixed (Dirichlet) nodes from the operator and push their known
//   pressures into the right-hand side (assemble_rhs below). Concretely:
//     * Fixed   row i:  y_i = x_i                       (trivial identity block)
//     * Interior row i: y_i = (sum incident G) x_i
//                              - sum over INTERIOR neighbors j of G_ij x_j
//       (a segment to a FIXED neighbor still adds G to the diagonal, but its
//        off-diagonal term is a CONSTANT and lives in b, not in L). Because both
//        endpoints of every kept off-diagonal are interior, L_ij = L_ji = -G:
//        the interior block is symmetric, and diagonally dominant with at least
//        one boundary-touching row, hence SPD -> CG is valid and converges.
//   `G` is the precomputed per-segment conductance so the physics (coronary.h)
//   is evaluated once per solve, not once per SpMV.
void spmv(const Network& net, const std::vector<double>& G,
          const std::vector<double>& x, std::vector<double>& y) {
    const int N = net.n_nodes;
    // Start each interior row at 0; fixed rows are the identity.
    for (int i = 0; i < N; ++i) y[i] = net.is_fixed[i] ? x[i] : 0.0;

    for (int s = 0; s < net.n_segs; ++s) {
        const int a = net.seg_a[s], b = net.seg_b[s];
        const double g = G[s];
        const bool fa = net.is_fixed[a] != 0, fb = net.is_fixed[b] != 0;
        // Diagonal: every incident segment adds its conductance to an interior
        // node's diagonal, regardless of whether the far end is fixed.
        if (!fa) y[a] += g * x[a];
        if (!fb) y[b] += g * x[b];
        // Off-diagonal: ONLY kept when BOTH endpoints are interior (so L stays
        // symmetric). A fixed neighbor's contribution is constant -> goes to b.
        if (!fa && !fb) {
            y[a] -= g * x[b];
            y[b] -= g * x[a];
        }
    }
}

// Conjugate Gradient for L p = rhs. L is SPD once at least one row is pinned.
//   Returns iterations used; writes final residual norm to *resid_out.
//   Textbook CG (Shewchuk): r = b - L x; p = r; loop { a = rr/(pLp); x += a p;
//   r -= a Lp; b = r'r'/rr; p = r + b p }. All reductions are plain serial sums
//   here; the GPU mirrors this with block reductions (see kernels.cu).
int cg_solve(const Network& net, const std::vector<double>& G,
             const std::vector<double>& rhs, std::vector<double>& x,
             double tol, int max_iter, double* resid_out) {
    const int N = net.n_nodes;
    std::vector<double> r(N), pvec(N), Lp(N);

    // r0 = b - L x0  (x0 is the incoming guess, typically previous solution)
    spmv(net, G, x, Lp);
    double rr = 0.0, bnorm = 0.0;
    for (int i = 0; i < N; ++i) {
        r[i]    = rhs[i] - Lp[i];
        pvec[i] = r[i];
        rr     += r[i] * r[i];
        bnorm  += rhs[i] * rhs[i];
    }
    const double thresh = tol * tol * (bnorm > 0.0 ? bnorm : 1.0);  // relative stop

    int it = 0;
    for (; it < max_iter; ++it) {
        if (rr <= thresh) break;                 // converged
        spmv(net, G, pvec, Lp);                  // Lp = L p
        double pLp = 0.0;
        for (int i = 0; i < N; ++i) pLp += pvec[i] * Lp[i];
        const double alpha = rr / pLp;           // exact line-search step
        double rr_new = 0.0;
        for (int i = 0; i < N; ++i) {
            x[i] += alpha * pvec[i];             // advance the solution
            r[i] -= alpha * Lp[i];               // update the residual
            rr_new += r[i] * r[i];
        }
        const double beta = rr_new / rr;         // Fletcher-Reeves ratio
        for (int i = 0; i < N; ++i) pvec[i] = r[i] + beta * pvec[i];
        rr = rr_new;
    }
    if (resid_out) *resid_out = std::sqrt(rr);
    return it;
}

// Build the right-hand side b for the current network, consistent with the
// boundary-eliminated operator in spmv():
//   * Fixed   row i: b_i = fixed_p[i]   (so the identity row gives p_i = fixed_p)
//   * Interior row i: b_i = sum over incident segments to a FIXED neighbor j of
//                     G_ij * fixed_p[j]   (the eliminated Dirichlet coupling).
// Interior nodes with no fixed neighbor get b_i = 0 (no injected flow). Because
// b depends on the conductances G, it must be rebuilt every autoregulation
// iteration -- so it takes G, unlike the topology-only original stub.
void assemble_rhs(const Network& net, const std::vector<double>& G,
                  std::vector<double>& b) {
    b.assign(net.n_nodes, 0.0);
    for (int i = 0; i < net.n_nodes; ++i)
        if (net.is_fixed[i]) b[i] = net.fixed_p[i];

    for (int s = 0; s < net.n_segs; ++s) {
        const int a = net.seg_a[s], b_ = net.seg_b[s];
        const double g = G[s];
        const bool fa = net.is_fixed[a] != 0, fb = net.is_fixed[b_] != 0;
        // Edge from interior a to fixed b_: adds G*fixed_p[b_] to interior row a.
        if (!fa && fb) b[a] += g * net.fixed_p[b_];
        // Edge from interior b_ to fixed a: adds G*fixed_p[a] to interior row b_.
        if (fa && !fb) b[b_] += g * net.fixed_p[a];
    }
}

}  // namespace

// ===========================================================================
// SECTION 4 -- The full outer autoregulation loop (the public entry point)
// ===========================================================================
void solve_cpu(Network& net, int n_autoreg, double cg_tol, int cg_max_iter,
               Solution& out) {
    const int N = net.n_nodes, S = net.n_segs;
    out.p.assign(N, 0.0);
    out.q.assign(S, 0.0);

    // Warm start: initialize interior pressures to the midpoint of the fixed
    // boundary pressures so CG converges in fewer iterations. (Any guess is fine
    // for correctness; this just mirrors the GPU init so both take equal steps.)
    double pmax = 0.0, pmin = 1e300;
    for (int i = 0; i < N; ++i) if (net.is_fixed[i]) {
        if (net.fixed_p[i] > pmax) pmax = net.fixed_p[i];
        if (net.fixed_p[i] < pmin) pmin = net.fixed_p[i];
    }
    const double pmid = 0.5 * (pmax + pmin);
    for (int i = 0; i < N; ++i) out.p[i] = net.is_fixed[i] ? net.fixed_p[i] : pmid;

    std::vector<double> G(S), b;

    // Autoregulation feedback constants (documented in coronary.h).
    const double gain = 0.20, rmin = 4.0, rmax = 40.0;

    for (int k = 0; k < n_autoreg; ++k) {
        // (a) conductances from current radii (r^4 law + Fahraeus-Lindqvist)
        for (int s = 0; s < S; ++s)
            G[s] = coronary::segment_conductance(net.seg_r[s], net.seg_len[s], net.hct);

        // (a') rebuild b from the (radius-dependent) fixed-neighbor coupling
        assemble_rhs(net, G, b);

        // (b) solve L p = b (sparse SPD) with CG, warm-started from previous p
        out.cg_iters = cg_solve(net, G, b, out.p, cg_tol, cg_max_iter, &out.cg_resid);

        // (c) flows implied by the new pressures
        compute_flows(net, out.p, out.q);

        // Record the FIRST (cold-start) solve's cost + pre-autoregulation
        // perfusion so the demo can report both honestly.
        if (k == 0) {
            out.cg_iters_first = out.cg_iters;
            out.perfusion_first = inlet_perfusion(net, out.q);
        }

        // (d) autoregulate radii toward the metabolic target -- EXCEPT on the
        //     final pass, so the returned (p,q) are self-consistent with radii.
        if (k + 1 < n_autoreg) {
            for (int s = 0; s < S; ++s)
                net.seg_r[s] = coronary::autoregulate_radius(
                    net.seg_r[s], out.q[s], net.seg_target[s], gain, rmin, rmax);
        }
    }
}
