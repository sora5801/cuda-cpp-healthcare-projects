// ===========================================================================
// src/reference_cpu.cpp  --  Trusted serial likelihood baseline + data loader
// ---------------------------------------------------------------------------
// Project 3.9 : Phylogenetic Likelihood / Tree Inference
//
// ROLE
//   (1) load_problem(): parse the tiny text dataset (format in data/README.md):
//       an alignment, the K2P kappa, and a set of candidate trees.
//   (2) score_trees_cpu(): the obviously-correct serial computation the GPU
//       kernel is verified against -- it loops over trees and sites, calling the
//       SHARED site_log_likelihood() from felsenstein.h, and accumulates each
//       tree's total log-likelihood in fixed-point integers (so the sum is the
//       exact same integer the GPU's atomicAdd produces).
//   (3) best_tree_index(): pick the maximum-likelihood tree deterministically.
//
//   Compiled by the host C++ compiler only (no CUDA). See reference_cpu.h.
//
// READ THIS AFTER: reference_cpu.h, felsenstein.h. Compare against kernels.cu
// (the GPU twin that runs the SAME site_log_likelihood per thread).
// ===========================================================================
#include "reference_cpu.h"

#include <cctype>      // std::toupper
#include <fstream>
#include <sstream>
#include <stdexcept>

// ---------------------------------------------------------------------------
// base_to_state: map a DNA character to a state index used everywhere.
//   A->0 C->1 G->2 T->3 ; '-', 'N', '?' and anything unknown -> PHYLO_GAP.
//   The A,C,G,T = 0,1,2,3 ordering is load-bearing: felsenstein.h's
//   is_transition() relies on purines {A,G} being the even indices {0,2}.
// ---------------------------------------------------------------------------
static unsigned char base_to_state(char c) {
    switch (std::toupper(static_cast<unsigned char>(c))) {
        case 'A': return 0;
        case 'C': return 1;
        case 'G': return 2;
        case 'T': case 'U': return 3;   // U (RNA) treated as T
        default:  return PHYLO_GAP;     // '-', 'N', '?', gaps, ambiguity codes
    }
}

// ---------------------------------------------------------------------------
// next_content_line: read the next non-blank, non-'#'-comment line. The format
// is line-oriented but tolerant of blank/comment lines so the demo input stays
// human-editable. A trailing CR is stripped so Windows-edited files parse anywhere.
// ---------------------------------------------------------------------------
static bool next_content_line(std::istream& in, std::string& line) {
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const std::size_t p = line.find_first_not_of(" \t");
        if (p == std::string::npos) continue;          // all whitespace -> skip
        if (line[p] == '#') continue;                  // comment line   -> skip
        return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// load_problem: parse the dataset. GRAMMAR (see data/README.md for the worked
// example). Counts live on their own lines; whitespace separates fields.
//
//   n_taxa n_sites n_trees kappa
//   <name_0> <sequence_0>            (n_taxa of these; sequence length == n_sites)
//   ...
//   # one block per tree:
//   <tree_label>
//   <n_internal>                     (== n_taxa - 1)
//   left right t_left t_right        (n_internal of these, POST-ORDER, root last)
//   ...
// ---------------------------------------------------------------------------
PhyloProblem load_problem(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open problem file: " + path);

    std::string line;
    if (!next_content_line(in, line))
        throw std::runtime_error("empty problem file: " + path);

    PhyloProblem prob;
    int n_taxa = 0, n_sites = 0, n_trees = 0;
    {
        std::istringstream hs(line);
        if (!(hs >> n_taxa >> n_sites >> n_trees >> prob.kappa))
            throw std::runtime_error("bad header (expected 'n_taxa n_sites n_trees kappa') in " + path);
    }
    if (n_taxa < 3 || n_sites < 1 || n_trees < 1)
        throw std::runtime_error("degenerate sizes in header of " + path);

    // --- read the alignment (n_taxa rows), storing it COLUMN-MAJOR ----------
    prob.align.n_taxa  = n_taxa;
    prob.align.n_sites = n_sites;
    prob.align.names.resize(n_taxa);
    prob.align.data.assign(static_cast<std::size_t>(n_sites) * n_taxa, PHYLO_GAP);
    for (int r = 0; r < n_taxa; ++r) {
        if (!next_content_line(in, line))
            throw std::runtime_error("not enough sequences in " + path);
        std::istringstream rs(line);
        std::string name, seq;
        if (!(rs >> name >> seq))
            throw std::runtime_error("bad sequence line (need '<name> <seq>') in " + path);
        if (static_cast<int>(seq.size()) != n_sites)
            throw std::runtime_error("sequence '" + name + "' length " +
                std::to_string(seq.size()) + " != n_sites " + std::to_string(n_sites));
        prob.align.names[r] = name;
        // Scatter this row into column-major storage: site j, taxon r.
        for (int j = 0; j < n_sites; ++j)
            prob.align.data[static_cast<std::size_t>(j) * n_taxa + r] = base_to_state(seq[j]);
    }

    // --- read the candidate trees -------------------------------------------
    const int n_internal_expected = n_taxa - 1;   // rooted binary tree invariant
    prob.trees.resize(n_trees);
    for (int t = 0; t < n_trees; ++t) {
        if (!next_content_line(in, line))
            throw std::runtime_error("missing label for tree " + std::to_string(t) + " in " + path);
        {   // the label is the first whitespace token on its line
            std::istringstream ls(line);
            ls >> prob.trees[t].label;
        }
        if (!next_content_line(in, line))
            throw std::runtime_error("missing n_internal for tree " + prob.trees[t].label);
        int n_internal = 0;
        { std::istringstream ns(line); ns >> n_internal; }
        if (n_internal != n_internal_expected)
            throw std::runtime_error("tree '" + prob.trees[t].label + "' has " +
                std::to_string(n_internal) + " internal nodes, expected " +
                std::to_string(n_internal_expected));

        prob.trees[t].nodes.resize(n_internal);
        for (int k = 0; k < n_internal; ++k) {
            if (!next_content_line(in, line))
                throw std::runtime_error("not enough node lines for tree " + prob.trees[t].label);
            std::istringstream es(line);
            PhyloNode nd{};
            if (!(es >> nd.left >> nd.right >> nd.t_left >> nd.t_right))
                throw std::runtime_error("bad node line (need 'left right t_left t_right') in tree " +
                                         prob.trees[t].label);
            // Validate child indices: a child is either a leaf (0..n_taxa-1) or an
            // EARLIER internal node (post-order => index n_taxa..n_taxa+k-1). This
            // guard is what makes the single forward sweep in felsenstein.h safe.
            const int max_child = n_taxa + k - 1;
            if (nd.left < 0 || nd.left > max_child || nd.right < 0 || nd.right > max_child)
                throw std::runtime_error("tree '" + prob.trees[t].label +
                    "' node " + std::to_string(k) + " references a not-yet-computed child "
                    "(post-order violated)");
            prob.trees[t].nodes[k] = nd;
        }
    }
    return prob;
}

// ---------------------------------------------------------------------------
// score_trees_cpu: the serial driver. For each tree, sum the per-site log-
// likelihood (felsenstein.h) in FIXED-POINT integers, then convert back to a
// double. Using the integer path here -- identical to the GPU's atomic-integer
// reduction -- is what makes the CPU and GPU totals match EXACTLY (THEORY
// "verify correctness"), not just within a floating tolerance.
//   Complexity: O(n_trees * n_sites * n_taxa) -- the n_sites factor is the
//   embarrassingly-parallel dimension the GPU attacks (one thread per site).
// ---------------------------------------------------------------------------
void score_trees_cpu(const PhyloProblem& prob, std::vector<double>& tree_lnL) {
    const int n_taxa  = prob.align.n_taxa;
    const int n_sites = prob.align.n_sites;
    tree_lnL.assign(prob.trees.size(), 0.0);

    // Scratch CLV buffer reused across sites: one NSTATES vector per node
    // (leaves + internal). Allocated once to keep the inner loop allocation-free.
    const int n_internal = n_taxa - 1;
    std::vector<double> clv(static_cast<std::size_t>(n_taxa + n_internal) * PHYLO_NSTATES);

    for (std::size_t t = 0; t < prob.trees.size(); ++t) {
        const CandidateTree& tree = prob.trees[t];
        long long fixed_total = 0;   // fixed-point accumulator (integer adds commute)
        for (int j = 0; j < n_sites; ++j) {
            const unsigned char* column = &prob.align.data[static_cast<std::size_t>(j) * n_taxa];
            const double site_lnL = site_log_likelihood(
                column, tree.nodes.data(), n_internal, n_taxa, prob.kappa, clv.data());
            fixed_total += to_fixed(site_lnL);     // round-to-fixed, then integer add
        }
        tree_lnL[t] = from_fixed(fixed_total);     // back to a floating lnL for display
    }
}

// ---------------------------------------------------------------------------
// best_tree_index: argmax of the log-likelihoods, ties broken by lower index so
// the chosen tree is deterministic (matters for reproducible stdout).
// ---------------------------------------------------------------------------
int best_tree_index(const std::vector<double>& tree_lnL) {
    int best = 0;
    for (int i = 1; i < static_cast<int>(tree_lnL.size()); ++i)
        if (tree_lnL[i] > tree_lnL[best]) best = i;   // strictly greater => keep first on ties
    return best;
}
