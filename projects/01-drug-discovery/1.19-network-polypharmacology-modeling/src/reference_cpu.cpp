// ===========================================================================
// src/reference_cpu.cpp  --  The trusted serial TransE baseline + data loader
// ---------------------------------------------------------------------------
// Project 1.19 : Network / Polypharmacology Modeling
//
// ROLE
//   (1) load_knowledge_graph(): parse the tiny text dataset (data/README.md
//       format) into a KnowledgeGraph (head, relation, n tail embeddings).
//   (2) transe_score_cpu(): the obviously-correct serial scorer the GPU kernel is
//       verified against. It loops over candidate tails and calls the SHARED
//       transe_score() from transe.h -- the exact same function the GPU thread
//       calls -- so CPU and GPU agree bit-for-bit (see ../THEORY.md "verify").
//
//   Compiled by the host C++ compiler only (no CUDA). See reference_cpu.h.
//
// READ THIS AFTER: reference_cpu.h, transe.h. Compare against kernels.cu (the
// GPU twin), whose kernel body is the same loop body run by one thread per tail.
// ===========================================================================
#include "reference_cpu.h"
#include "transe.h"          // transe_score() -- shared host/device per-tail math

#include <fstream>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
// read_n_floats: read exactly `count` whitespace-separated floats into dst.
//   A small local helper so the loader stays readable. Throws if the stream runs
//   dry early (a truncated/corrupt file) so the demo fails loudly, not silently.
// ---------------------------------------------------------------------------
static void read_n_floats(std::ifstream& in, std::vector<float>& dst, int count,
                          const std::string& path) {
    dst.resize(static_cast<std::size_t>(count));
    for (int k = 0; k < count; ++k) {
        if (!(in >> dst[static_cast<std::size_t>(k)]))
            throw std::runtime_error("unexpected end of data while reading floats in " + path);
    }
}

KnowledgeGraph load_knowledge_graph(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open knowledge-graph file: " + path);

    // line 1: header "<n> <dim>"
    int n = 0, dim = 0;
    if (!(in >> n >> dim))
        throw std::runtime_error("bad header (expected '<n> <dim>') in " + path);
    if (n <= 0 || dim <= 0)
        throw std::runtime_error("non-positive n or dim in " + path);

    KnowledgeGraph kg;
    kg.n = n;
    kg.dim = dim;

    // line 2: the head (drug) embedding; line 3: the relation embedding.
    read_n_floats(in, kg.head, dim, path);
    read_n_floats(in, kg.relation, dim, path);

    // line 4: "<n_true> idx0 idx1 ..." ground-truth target indices (for report).
    int n_true = 0;
    if (!(in >> n_true) || n_true < 0)
        throw std::runtime_error("bad ground-truth count in " + path);
    kg.true_targets.resize(static_cast<std::size_t>(n_true));
    for (int j = 0; j < n_true; ++j) {
        if (!(in >> kg.true_targets[static_cast<std::size_t>(j)]))
            throw std::runtime_error("unexpected end while reading true targets in " + path);
    }

    // remaining n lines: the candidate tail embeddings, flattened row-major.
    read_n_floats(in, kg.tails, n * dim, path);
    return kg;
}

// ---------------------------------------------------------------------------
// transe_score_cpu: score every candidate tail serially.
//   For candidate j we pass the head, the relation, and a pointer to tail j's
//   row (kg.tails + j*dim) into the SHARED transe_score(). Because this is the
//   identical function the GPU thread calls, the two results match exactly.
//   Complexity: O(n * dim) -- a flat double loop, no allocation in the hot path.
// ---------------------------------------------------------------------------
void transe_score_cpu(const KnowledgeGraph& kg, std::vector<float>& score) {
    score.assign(static_cast<std::size_t>(kg.n), 0.0f);
    const float* h = kg.head.data();
    const float* r = kg.relation.data();
    for (int j = 0; j < kg.n; ++j) {
        // Pointer to the start of candidate tail j's embedding row.
        const float* t = kg.tails.data() + static_cast<std::size_t>(j) * kg.dim;
        score[static_cast<std::size_t>(j)] = transe_score(h, r, t, kg.dim);
    }
}
