// ===========================================================================
// src/reference_cpu.cpp  --  The plain-C++ baseline we trust (loader + MI + DPI)
// ---------------------------------------------------------------------------
// Project 6.13 : Gene Regulatory Network Inference (ARACNE: MI + DPI)
//
// ROLE IN THE PROJECT
//   The "ground truth" the GPU result is checked against. It is written to be
//   OBVIOUSLY correct -- plain nested loops, no parallelism -- so that when the
//   GPU and CPU agree we believe the GPU. It reuses grn.h's mi_from_joint() and
//   discretize_value() so the ACTUAL MATH is identical to the kernel; only the
//   loop structure differs. Compiled by the host C++ compiler (no CUDA here).
//
// FOUR FUNCTIONS, matching the pipeline in reference_cpu.h:
//   load_expression()   parse the tiny text dataset (data/README.md format)
//   discretize_matrix() raw expression -> per-gene equal-width bins
//   mi_matrix_cpu()      every gene pair -> mutual information (nats)
//   dpi_prune_cpu()      Data Processing Inequality -> direct-edge mask
//
// READ THIS AFTER: reference_cpu.h, grn.h. Compare against kernels.cu (the twin).
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

// ---------------------------------------------------------------------------
// load_expression : parse "<G> <S>" then G lines of "<name> v0 .. v(S-1)".
//   We validate the header and every row length so a malformed sample fails
//   loudly (a silent short read would corrupt the MI silently). Values are
//   stored as double in row-major [G*S] layout (gene g, sample s -> g*S + s).
// ---------------------------------------------------------------------------
GrnData load_expression(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open expression file: " + path);

    int G = 0, S = 0;
    if (!(in >> G >> S))
        throw std::runtime_error("bad header (expected '<n_genes> <n_samples>') in " + path);
    if (G <= 0 || S <= 0)
        throw std::runtime_error("non-positive matrix dimensions in " + path);

    GrnData d;
    d.n_genes = G;
    d.n_samples = S;
    d.expr.resize(static_cast<std::size_t>(G) * S);
    d.disc.resize(static_cast<std::size_t>(G) * S);
    d.gene_names.resize(G);

    // Read row by row. The first token on each row is the gene's name (a label
    // for the report); the next S tokens are its expression across the samples.
    for (int g = 0; g < G; ++g) {
        if (!(in >> d.gene_names[g]))
            throw std::runtime_error("unexpected end of data (gene name) in " + path);
        for (int s = 0; s < S; ++s) {
            double v;
            if (!(in >> v))
                throw std::runtime_error("unexpected end of data (values) in " + path);
            d.expr[static_cast<std::size_t>(g) * S + s] = v;
        }
    }
    return d;
}

// ---------------------------------------------------------------------------
// discretize_matrix : per-gene equal-width binning into [0, N_BINS).
//   For each gene we first scan its S values for [min,max], then map every value
//   through grn.h::discretize_value(). Per-gene ranges (rather than one global
//   range) mean each gene uses its full dynamic range -- the standard ARACNE
//   choice. This is the SAME routine the GPU runs on-device, so the two `disc`
//   matrices are bit-identical and the joint counts downstream match exactly.
//   Complexity: O(G * S).
// ---------------------------------------------------------------------------
void discretize_matrix(GrnData& data) {
    const int G = data.n_genes, S = data.n_samples;
    for (int g = 0; g < G; ++g) {
        const double* row = &data.expr[static_cast<std::size_t>(g) * S];
        double lo = row[0], hi = row[0];
        for (int s = 1; s < S; ++s) {           // find this gene's [min,max]
            if (row[s] < lo) lo = row[s];
            if (row[s] > hi) hi = row[s];
        }
        for (int s = 0; s < S; ++s) {           // map each value to a bin
            int bin = discretize_value(row[s], lo, hi);
            data.disc[static_cast<std::size_t>(g) * S + s] = static_cast<uint8_t>(bin);
        }
    }
}

// ---------------------------------------------------------------------------
// mi_matrix_cpu : dense symmetric MI matrix over all gene pairs.
//   For each unordered pair (i<j) we accumulate the B x B joint histogram over
//   the S samples, call mi_from_joint() (the shared core), and store the result
//   symmetrically. The diagonal is left 0 (self-MI is uninteresting here).
//   Complexity: O(G^2 * S) counting + O(G^2 * B^2) MI evaluation -- the O(G^2)
//   pair count is exactly the bottleneck the GPU parallelizes (THEORY sec GPU).
// ---------------------------------------------------------------------------
void mi_matrix_cpu(const GrnData& data, std::vector<double>& mi) {
    const int G = data.n_genes, S = data.n_samples;
    mi.assign(static_cast<std::size_t>(G) * G, 0.0);

    int joint[JOINT_CELLS];   // B*B counts for the current pair (reused per pair)
    for (int i = 0; i < G; ++i) {
        const uint8_t* di = &data.disc[static_cast<std::size_t>(i) * S];
        for (int j = i + 1; j < G; ++j) {
            const uint8_t* dj = &data.disc[static_cast<std::size_t>(j) * S];

            // Zero the joint table, then tally one increment per sample: the
            // sample's (bin of gene i, bin of gene j) cell.
            for (int c = 0; c < JOINT_CELLS; ++c) joint[c] = 0;
            for (int s = 0; s < S; ++s)
                joint[di[s] * N_BINS + dj[s]] += 1;

            double val = mi_from_joint(joint, S);         // shared core (grn.h)
            mi[static_cast<std::size_t>(i) * G + j] = val; // symmetric fill
            mi[static_cast<std::size_t>(j) * G + i] = val;
        }
    }
}

// ---------------------------------------------------------------------------
// dpi_prune_cpu : the Data Processing Inequality filter.
//   Start by keeping every above-threshold edge, then for each triangle (i,j,k)
//   test whether edge (i,j) is the WEAKEST of the three by more than `tolerance`:
//       I(i,j) < I(i,k) - tol  AND  I(i,j) < I(j,k) - tol
//   If so, (i,j) is most plausibly INDIRECT (mediated through k) and is removed.
//   We evaluate all removals against the ORIGINAL MI matrix (not a mutating mask)
//   so the outcome does not depend on iteration order -> fully deterministic and
//   identical to the GPU. Complexity: O(G^3).
// ---------------------------------------------------------------------------
void dpi_prune_cpu(const std::vector<double>& mi, int n_genes,
                   double mi_threshold, double tolerance,
                   std::vector<uint8_t>& keep) {
    const int G = n_genes;
    keep.assign(static_cast<std::size_t>(G) * G, 0);

    // Seed: keep an edge if it is above the MI significance threshold.
    for (int i = 0; i < G; ++i)
        for (int j = i + 1; j < G; ++j) {
            double w = mi[static_cast<std::size_t>(i) * G + j];
            uint8_t k = (w > mi_threshold) ? 1 : 0;
            keep[static_cast<std::size_t>(i) * G + j] = k;
            keep[static_cast<std::size_t>(j) * G + i] = k;
        }

    // DPI: prune (i,j) if some k makes it the strictly-weakest triangle edge.
    for (int i = 0; i < G; ++i) {
        for (int j = i + 1; j < G; ++j) {
            if (!keep[static_cast<std::size_t>(i) * G + j]) continue;  // already gone
            double wij = mi[static_cast<std::size_t>(i) * G + j];
            for (int k = 0; k < G; ++k) {
                if (k == i || k == j) continue;
                double wik = mi[static_cast<std::size_t>(i) * G + k];
                double wjk = mi[static_cast<std::size_t>(j) * G + k];
                if (wij < wik - tolerance && wij < wjk - tolerance) {
                    keep[static_cast<std::size_t>(i) * G + j] = 0;     // indirect
                    keep[static_cast<std::size_t>(j) * G + i] = 0;
                    break;
                }
            }
        }
    }
}
