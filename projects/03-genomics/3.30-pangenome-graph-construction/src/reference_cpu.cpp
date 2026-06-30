// ===========================================================================
// src/reference_cpu.cpp  --  Loader, term builder, serial layout reference
// ---------------------------------------------------------------------------
// Project 3.30 : Pangenome Graph Construction
// Compiled by the host compiler only. The per-term physics lives in layout.h.
//
// This file is the TRUSTED baseline. It does, serially, exactly what kernels.cu
// does in parallel:
//   per sweep:
//     1) ZERO the per-node fixed-point force accumulators.
//     2) For every term, compute LO_term_displacement() for endpoint i and
//        atomic-add (here: just add) the FIXED-POINT force to nodes i and j.
//     3) Apply: x[k] += LO_from_fixed(force[k]).
//   The fixed-point quantisation (layout.h) is what makes step 2 commute, so the
//   serial sum here and the GPU's atomic sum are bit-for-bit identical.
// ===========================================================================
#include "reference_cpu.h"

#include <algorithm>   // std::fill
#include <cmath>       // std::pow
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <utility>     // std::pair, std::make_pair

// ---------------------------------------------------------------------------
// load_pangenome: parse the tiny teaching format. See data/README.md.
//   Line 1 : "N P"            -- N nodes, P genome paths.
//   Line 2 : N integers       -- node lengths in base pairs.
//   Next P lines: "L  id0 id1 ... id{L-1}"  -- a path: its length then node ids.
//   '#' comment lines and trailing comments are stripped so the sample can be
//   annotated. We parse token-by-token (after comment removal) for robustness.
// ---------------------------------------------------------------------------
Pangenome load_pangenome(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open pangenome file: " + path);

    std::vector<std::string> tokens;   // all non-comment whitespace tokens, in order
    std::string line;
    while (std::getline(in, line)) {
        const std::size_t hash = line.find('#');           // strip trailing comment
        if (hash != std::string::npos) line.erase(hash);
        std::istringstream ls(line);
        std::string t;
        while (ls >> t) tokens.push_back(t);
    }

    std::size_t pos = 0;
    auto next_int = [&](const char* what) -> long {
        if (pos >= tokens.size())
            throw std::runtime_error(std::string("pangenome truncated: expected ") + what);
        return std::stol(tokens[pos++]);
    };

    Pangenome g;
    g.num_nodes = static_cast<int>(next_int("N (node count)"));
    const int P = static_cast<int>(next_int("P (path count)"));
    if (g.num_nodes <= 0 || P <= 0)
        throw std::runtime_error("pangenome header must have N>0 and P>0");

    g.node_len.resize(g.num_nodes);
    for (int k = 0; k < g.num_nodes; ++k)
        g.node_len[k] = static_cast<int>(next_int("node length"));

    g.paths.resize(P);
    for (int p = 0; p < P; ++p) {
        const int L = static_cast<int>(next_int("path length"));
        if (L <= 0) throw std::runtime_error("path length must be > 0");
        g.paths[p].resize(L);
        for (int s = 0; s < L; ++s) {
            const int id = static_cast<int>(next_int("node id on path"));
            if (id < 0 || id >= g.num_nodes)
                throw std::runtime_error("path references a node id out of range");
            g.paths[p][s] = id;
        }
    }
    return g;
}

// ---------------------------------------------------------------------------
// init_layout: deterministic starting coordinates.
//   We walk path 0 (the "reference" genome) and place each node it visits at the
//   running base-pair offset (cumulative node length), so the reference genome is
//   laid out left-to-right exactly as it reads -- a natural frame. Nodes not on
//   path 0, and repeat visits, use "first placement wins" so the map is
//   single-valued and order-independent. Leftover nodes are appended in id order
//   after the reference span. This mirrors ODGI's habit of seeding the layout
//   from an existing path order.
// ---------------------------------------------------------------------------
void init_layout(const Pangenome& g, std::vector<double>& x) {
    x.assign(g.num_nodes, 0.0);
    std::vector<char> placed(g.num_nodes, 0);   // 0/1: does this node have a coord yet?

    double offset = 0.0;                          // running base-pair position
    if (!g.paths.empty()) {
        for (int id : g.paths[0]) {
            if (!placed[id]) {
                x[id] = offset;                   // first sighting of this node: pin it
                placed[id] = 1;
            }
            offset += g.node_len[id];             // advance by this node's length
        }
    }
    for (int k = 0; k < g.num_nodes; ++k) {       // append never-on-path-0 nodes
        if (!placed[k]) {
            x[k] = offset;
            offset += g.node_len[k];
            placed[k] = 1;
        }
    }
}

// ---------------------------------------------------------------------------
// build_problem: turn the graph topology into stress terms.
//   For each path and each start index s, look ahead up to `hops` steps. The term
//   between node at s and node at s+h has target separation = the bp travelled
//   from the start of node s to the start of node s+h (sum of intervening node
//   lengths). Weight = 1/target^2 (ODGI's weighting: short distances matter most).
//
//   The same unordered pair {a,b} may arise from several paths or hop counts. We
//   MERGE duplicates by keeping the SMALLEST target distance (the tightest
//   constraint) in an ORDERED std::map keyed by the pair, so the emitted term
//   array is identical regardless of path order -- which the GPU then consumes in
//   that fixed order, keeping CPU and GPU in lockstep. SMACOF needs no schedule.
// ---------------------------------------------------------------------------
LayoutProblem build_problem(const Pangenome& g, int hops, int iters) {
    LayoutProblem prob;
    prob.iters = iters;
    init_layout(g, prob.init_x);

    std::map<std::pair<int, int>, double> best;   // (lo,hi) -> smallest target dist
    for (const auto& path : g.paths) {
        const int L = static_cast<int>(path.size());
        for (int s = 0; s < L; ++s) {
            double dist = 0.0;                      // bp travelled from start-of(s)
            for (int h = 1; h <= hops && s + h < L; ++h) {
                dist += g.node_len[path[s + h - 1]];   // add the node we step over
                const int a = path[s];
                const int b = path[s + h];
                if (a == b) continue;               // a path self-loop: no term
                if (dist <= 0.0) continue;          // guard a zero-length target
                const int lo = a < b ? a : b;
                const int hi = a < b ? b : a;
                auto key = std::make_pair(lo, hi);
                auto it  = best.find(key);
                if (it == best.end() || dist < it->second) best[key] = dist;
            }
        }
    }

    prob.terms.reserve(best.size());               // emit in sorted (deterministic) order
    double w_max = 0.0;                             // largest weight = 1/d_min^2
    for (const auto& kv : best) {
        LayoutTerm t;
        t.i        = kv.first.first;
        t.j        = kv.first.second;
        t.target_d = kv.second;
        t.weight   = 1.0 / (kv.second * kv.second);   // 1/d^2 (short distances matter most)
        prob.terms.push_back(t);
        if (t.weight > w_max) w_max = t.weight;
    }

    // NORMALISE weights so the largest is exactly 1.0. The Guttman ratio
    // numerator/denominator is INVARIANT to a global weight scale (both sums scale
    // together), so this changes nothing mathematically -- but it lifts the
    // SMALLEST weight from ~1e-6 to (d_min/d_max)^2, which the fixed-point
    // quantiser (layout.h) then represents with thousands of quanta instead of a
    // handful. Without this, the long-distance terms' weights would round to a few
    // coarse quanta and the division would lose precision.
    if (w_max > 0.0)
        for (LayoutTerm& t : prob.terms) t.weight /= w_max;

    return prob;
}

// ---------------------------------------------------------------------------
// compute_stress: the layout objective, E(x) = sum_terms w*(|x_i-x_j| - d)^2.
//   A fixed term order makes it deterministic; used both to report progress and
//   as a single scalar CPU/GPU cross-check.
// ---------------------------------------------------------------------------
double compute_stress(const LayoutProblem& p, const std::vector<double>& x) {
    double e = 0.0;
    for (const LayoutTerm& t : p.terms) {
        const double s   = x[t.i] - x[t.j];
        const double sa  = s < 0.0 ? -s : s;
        const double res = sa - t.target_d;
        e += t.weight * res * res;
    }
    return e;
}

// ---------------------------------------------------------------------------
// layout_cpu: the serial reference. Full-batch SMACOF (the Guttman transform),
//   done with the SAME fixed-point accumulation the GPU uses, so the two agree
//   exactly. Each sweep is a JACOBI update: all new positions are computed from
//   the sweep's STARTING positions, then applied together.
//     per sweep:
//       1) snapshot the current positions (read-only source for this sweep).
//       2) zero the per-node NUMERATOR and DENOMINATOR fixed-point accumulators.
//       3) for every term, add its Guttman numerator (LO_term_numerator) to BOTH
//          endpoints and its weight to BOTH denominators.
//       4) x[k] = numerator[k] / denominator[k]   (the weighted average).
//   Denominators sum only positive weights, so they never hit zero unless a node
//   has no terms (we then leave it fixed).
// ---------------------------------------------------------------------------
double layout_cpu(const LayoutProblem& p, std::vector<double>& x) {
    x = p.init_x;                                      // start from the shared init
    const int N = static_cast<int>(x.size());
    std::vector<long long> num(N);                    // per-node numerator (fixed-point)
    std::vector<long long> den(N);                    // per-node denominator (fixed-point)
    std::vector<double>    src(N);                    // sweep-start snapshot

    for (int it = 0; it < p.iters; ++it) {
        src = x;                                       // (1) freeze this sweep's source
        std::fill(num.begin(), num.end(), 0LL);        // (2) zero accumulators
        std::fill(den.begin(), den.end(), 0LL);

        // (3) Scatter each term's Guttman contribution onto BOTH endpoints. We
        //     quantise EACH contribution separately (matching the atomicAdds the
        //     kernel issues) so the rounding is bit-identical to the GPU.
        for (const LayoutTerm& t : p.terms) {
            const double ni = LO_term_numerator(src[t.i], src[t.j], t.target_d, t.weight);
            const double nj = LO_term_numerator(src[t.j], src[t.i], t.target_d, t.weight);
            num[t.i] += LO_to_fixed(ni);
            num[t.j] += LO_to_fixed(nj);
            den[t.i] += LO_to_fixed(t.weight);
            den[t.j] += LO_to_fixed(t.weight);
        }

        // (4) Apply: weighted average = numerator / denominator. A node with no
        //     terms (denominator 0) keeps its position.
        for (int k = 0; k < N; ++k)
            if (den[k] != 0) x[k] = LO_from_fixed(num[k]) / LO_from_fixed(den[k]);
    }
    return compute_stress(p, x);
}
