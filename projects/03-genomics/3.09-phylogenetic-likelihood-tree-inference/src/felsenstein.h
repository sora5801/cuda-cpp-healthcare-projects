// ===========================================================================
// src/felsenstein.h  --  Shared (host + device) phylogenetic likelihood core
// ---------------------------------------------------------------------------
// Project 3.9 : Phylogenetic Likelihood / Tree Inference
//
// WHAT THIS PROJECT COMPUTES
//   Given a fixed multiple-sequence alignment of DNA (n_taxa rows, n_sites
//   columns) and a candidate evolutionary TREE (topology + branch lengths), the
//   Felsenstein "pruning" recursion computes the LOG-LIKELIHOOD of that tree:
//   the log-probability that the observed sequences arose by mutation along the
//   tree's branches, under a substitution model. We score several candidate
//   trees and pick the one with the highest log-likelihood -- a maximum-
//   likelihood (ML) tree-selection step, the inner loop of RAxML/IQ-TREE/MrBayes.
//
//   The site likelihoods are INDEPENDENT across alignment columns, so the GPU
//   gives each site its own thread (kernels.cu). The PER-SITE math lives HERE as
//   __host__ __device__ inline functions, so the CPU reference (reference_cpu)
//   and the GPU kernel run BYTE-FOR-BYTE identical arithmetic -> verification is
//   exact in fixed-point and tight in floating point. (PATTERNS.md sec 2.)
//
// THE MODEL  (Kimura 2-parameter, "K2P")
//   DNA has 4 states {A,C,G,T}. Mutations are a continuous-time Markov process
//   on these 4 states. K2P distinguishes TRANSITIONS (purine<->purine A<->G, or
//   pyrimidine<->pyrimidine C<->T) from TRANSVERSIONS (the other 4 changes),
//   because transitions happen more often in real DNA. Over a branch of length t
//   (expected substitutions per site), the 4x4 transition-probability matrix
//   P(t) = exp(Q t) has a closed form (no numerical matrix exponential needed):
//       p0(t) = P(no change)        = 1/4 + 1/4 e^{-4 b t} + 1/2 e^{-2(a+b) t}
//       p1(t) = P(a transition)     = 1/4 + 1/4 e^{-4 b t} - 1/2 e^{-2(a+b) t}
//       p2(t) = P(a transversion)   = 1/4 - 1/4 e^{-4 b t}
//   where 'a' is the transition rate and 'b' the transversion rate (kappa = a/b
//   is the transition/transversion ratio; kappa=1 reduces K2P to Jukes-Cantor).
//   Each of A,C,G,T has two transversion partners and one transition partner, so
//   a row of P(t) is {p0 (diagonal), one p1, two p2}. We derive the right entry
//   per (from,to) pair with k2p_prob() below -- that single function is the only
//   place the model lives, shared by CPU and GPU.
//
//   Nucleotide order is fixed as A=0, C=1, G=2, T=3 everywhere (loader, kernel,
//   reference) so the transition/transversion classification is consistent.
//
// READ THIS BEFORE: reference_cpu.h, kernels.cuh. The derivation is in
// ../THEORY.md ("The math" and "The algorithm").
// ===========================================================================
#pragma once

#include <cmath>     // std::exp, std::log  (host); device uses the CUDA exp/log

// HD-macro idiom (PATTERNS.md sec 2): under nvcc, FELS_HD expands to the CUDA
// decorators so the SAME inline function compiles for BOTH the host and the
// device; under the plain host compiler it expands to nothing. Keep CUDA-only
// types (and __global__) OUT of this header so reference_cpu.cpp can include it.
#ifdef __CUDACC__
#define FELS_HD __host__ __device__
#else
#define FELS_HD
#endif

// --- model / problem-size constants ---------------------------------------
// Four DNA states. Kept as a named constant so the conditional-likelihood
// vectors and the partial-likelihood loops are self-documenting.
#define PHYLO_NSTATES 4

// A leaf observation is encoded as one of the four bases (0..3) or "gap/unknown"
// (PHYLO_GAP). A gap contributes a conditional likelihood of 1 for every state
// (it tells us nothing), which is exactly how real tools treat missing data.
#define PHYLO_GAP 4

// ---------------------------------------------------------------------------
// is_transition(from, to): true iff the substitution from->to is a TRANSITION
//   (A<->G are the purines = states 0,2 ; C<->T are the pyrimidines = 1,3).
//   Any other off-diagonal pair is a transversion. We compute it from the state
//   indices so there is no lookup table to get out of sync.
// ---------------------------------------------------------------------------
FELS_HD inline bool is_transition(int from, int to) {
    // Purines {A=0, G=2} have even index; pyrimidines {C=1, T=3} have odd index.
    // A transition keeps you within the same parity class AND changes the base.
    return (from != to) && ((from & 1) == (to & 1));
}

// ---------------------------------------------------------------------------
// k2p_prob(from, to, t, kappa): the (from->to) entry of the K2P transition
// matrix P(t) = exp(Q t), in CLOSED FORM (see the header comment for the math).
//   from, to : nucleotide indices 0..3 (A,C,G,T)
//   t        : branch length = expected substitutions per site (>= 0)
//   kappa    : transition/transversion rate ratio (kappa=1 => Jukes-Cantor)
//   returns  : a probability in (0,1]; the four 'to' values for a fixed 'from'
//              sum to 1 (it is a stochastic matrix row).
//
//   We parameterise so the TOTAL substitution rate is normalised to 1 (t is in
//   substitutions/site). With one transition partner and two transversion
//   partners per base, the mean rate is a + 2b; we set a + 2b such that the
//   expected number of substitutions over time t equals t. Concretely we use the
//   standard K2P rates a = kappa/(kappa+2), b = 1/(kappa+2) (so a+2b = 1), giving
//   the exponents below. This is exactly the parameterisation RAxML/PhyML use.
// ---------------------------------------------------------------------------
FELS_HD inline double k2p_prob(int from, int to, double t, double kappa) {
    // Rates normalised so a + 2b = 1 (one transition + two transversion targets).
    const double b = 1.0 / (kappa + 2.0);   // transversion rate
    const double a = kappa * b;             // transition rate = kappa * b
    // The two exponential terms that appear in the closed-form P(t).
    const double e_tv  = exp(-4.0 * b * t);             // e^{-4 b t}
    const double e_ts  = exp(-2.0 * (a + b) * t);       // e^{-2(a+b) t}
    if (from == to) {
        // Diagonal: probability the base is unchanged.
        return 0.25 + 0.25 * e_tv + 0.5 * e_ts;
    } else if (is_transition(from, to)) {
        // One transition partner.
        return 0.25 + 0.25 * e_tv - 0.5 * e_ts;
    } else {
        // Two transversion partners (each gets this same probability).
        return 0.25 - 0.25 * e_tv;
    }
}

// ---------------------------------------------------------------------------
// A compact tree node, post-order indexed. Leaves are taxa; internal nodes name
// their two children by index and carry the branch length to each child.
//
//   For node k (an INTERNAL node) we store:
//     left, right        : indices of the two child nodes
//     t_left, t_right     : branch lengths (subs/site) on the two child edges
//   Leaves (taxa) are nodes 0..n_taxa-1 and are NOT stored here; an internal
//   node references a leaf simply by giving a child index < n_taxa.
//
//   A binary unrooted tree on n_taxa leaves has n_taxa-2 internal nodes when
//   written as a rooted binary tree with the root collapsed; we store it ROOTED
//   for the pruning recursion (the likelihood of a reversible model is the same
//   wherever you root -- "pulley principle"), so there are n_taxa-1 internal
//   nodes, the last of which (highest index) is the root.
// ---------------------------------------------------------------------------
struct PhyloNode {
    int    left;       // child node index (a leaf if < n_taxa, else internal)
    int    right;      // child node index
    double t_left;     // branch length to the left child  (subs/site)
    double t_right;    // branch length to the right child (subs/site)
};

// ---------------------------------------------------------------------------
// leaf_clv(state, s): the conditional-likelihood value (CLV) of leaf-state
// `state` for nucleotide `s`. A leaf that we OBSERVED to be base b has CLV 1 for
// s==b and 0 otherwise (we are certain); a GAP has CLV 1 for every s (it is
// uninformative). This is the base case of the pruning recursion.
// ---------------------------------------------------------------------------
FELS_HD inline double leaf_clv(int state, int s) {
    if (state == PHYLO_GAP) return 1.0;          // missing data: all states allowed
    return (state == s) ? 1.0 : 0.0;             // observed base: a one-hot vector
}

// ---------------------------------------------------------------------------
// site_log_likelihood: the heart of the project -- Felsenstein's pruning
// recursion for ONE alignment column, returning ln L(site | tree, model).
//
//   column   : pointer to this site's n_taxa leaf states (0..3, or PHYLO_GAP),
//              i.e. one nucleotide per taxon at this alignment position.
//   nodes    : the internal nodes in POST-ORDER (children before parents), so a
//              single forward sweep visits every node only after its children.
//   n_internal : number of internal nodes (the root is nodes[n_internal-1]).
//   n_taxa   : number of leaves (taxa); also the index offset for internal CLVs.
//   kappa    : K2P transition/transversion ratio.
//   clv      : SCRATCH buffer of length (n_taxa + n_internal) * PHYLO_NSTATES.
//              The caller owns it; on the GPU each thread passes its own slice
//              (in registers/local memory) so threads never collide.
//
//   ALGORITHM (post-order pruning):
//     1. For each leaf l: its CLV is the one-hot/gap vector (leaf_clv).
//     2. For each internal node k (children already done): for each state s at k,
//          L_k[s] = ( sum_x P(s->x, t_left) * L_left[x] )
//                 * ( sum_y P(s->y, t_right) * L_right[y] )
//        i.e. the probability of the data below k GIVEN k is in state s, formed
//        by independently "pruning" the two subtrees and multiplying.
//     3. At the root, combine with the equilibrium base frequencies (1/4 each
//        under K2P) to get the site likelihood, and return its natural log.
//
//   COMPLEXITY: O(n_internal * NSTATES^2) multiply-adds per site = O(n_taxa)
//   work per site (NSTATES=4 is constant). Done for n_sites independent sites.
// ---------------------------------------------------------------------------
FELS_HD inline double site_log_likelihood(const unsigned char* column,
                                          const PhyloNode* nodes,
                                          int n_internal, int n_taxa,
                                          double kappa,
                                          double* clv) {
    // --- (1) base case: fill every LEAF's conditional-likelihood vector ------
    // Leaf l occupies clv[l*NSTATES .. l*NSTATES+3].
    for (int l = 0; l < n_taxa; ++l) {
        const int obs = column[l];                       // observed state at taxon l
        double* row = clv + l * PHYLO_NSTATES;
        for (int s = 0; s < PHYLO_NSTATES; ++s)
            row[s] = leaf_clv(obs, s);
    }

    // --- (2) recursion: sweep internal nodes in post-order -------------------
    // Internal node k is stored at nodes[k] and its CLV lives at slot
    // (n_taxa + k) in clv[]. Because the nodes are post-ordered, both children
    // already have their CLVs filled when we reach k.
    for (int k = 0; k < n_internal; ++k) {
        const PhyloNode nd = nodes[k];
        const double* Lc = clv + nd.left  * PHYLO_NSTATES;   // left  child CLV
        const double* Rc = clv + nd.right * PHYLO_NSTATES;   // right child CLV
        double* Lk = clv + (n_taxa + k) * PHYLO_NSTATES;     // this node's CLV (out)

        // For each possible state s at node k, marginalise over the child states.
        for (int s = 0; s < PHYLO_NSTATES; ++s) {
            double sum_left  = 0.0;   // sum_x P(s->x, t_left)  * Lc[x]
            double sum_right = 0.0;   // sum_y P(s->y, t_right) * Rc[y]
            for (int x = 0; x < PHYLO_NSTATES; ++x) {
                sum_left  += k2p_prob(s, x, nd.t_left,  kappa) * Lc[x];
                sum_right += k2p_prob(s, x, nd.t_right, kappa) * Rc[x];
            }
            // The two subtrees are conditionally independent given state s, so
            // the partial likelihood at k for state s is their product.
            Lk[s] = sum_left * sum_right;
        }
    }

    // --- (3) root: weight by equilibrium frequencies (1/4 each in K2P) -------
    const double* Lroot = clv + (n_taxa + (n_internal - 1)) * PHYLO_NSTATES;
    double site_L = 0.0;
    for (int s = 0; s < PHYLO_NSTATES; ++s)
        site_L += 0.25 * Lroot[s];                 // pi_s = 1/4 for all s under K2P

    // Return the natural log; guard the (degenerate) zero-likelihood case so we
    // never take log(0). With real data site_L > 0 always.
    return (site_L > 0.0) ? log(site_L) : -1.0e30;
}

// ---------------------------------------------------------------------------
// FIXED-POINT SCALING for a DETERMINISTIC total log-likelihood.
//   The GPU sums n_sites per-site log-likelihoods. A floating-point atomic sum
//   is NON-deterministic (add order varies between launches) and would not match
//   the CPU bit-for-bit (PATTERNS.md sec 3). So we convert each per-site lnL to a
//   FIXED-POINT integer (multiply by a large scale, round to nearest) and
//   accumulate those integers with atomicAdd on a 64-bit int -- integer adds
//   commute, so the total is reproducible AND identical to the CPU, which sums
//   the same integers. We divide back by the scale at the very end to report lnL.
//
//   Scale 1e6 keeps ~6 fractional digits; with |lnL_site| < ~50 and up to ~1e6
//   sites, the running total stays far inside the +-9.2e18 range of int64.
// ---------------------------------------------------------------------------
#define PHYLO_FIXED_SCALE 1000000.0   // 1e6: six decimal places of lnL preserved

// to_fixed: round a (negative) per-site log-likelihood to a fixed-point integer.
//   llround rounds half-away-from-zero IDENTICALLY on host and device, so the
//   integer is the same on both -> the summed total is bit-identical.
FELS_HD inline long long to_fixed(double lnL) {
    return (long long)llround(lnL * PHYLO_FIXED_SCALE);
}

// from_fixed: convert a fixed-point accumulator back to a floating lnL for display.
FELS_HD inline double from_fixed(long long fixed) {
    return (double)fixed / PHYLO_FIXED_SCALE;
}
