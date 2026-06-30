// ===========================================================================
// src/reference_cpu.cpp  --  Loader, query self-overlap, serial CPU screen
// ---------------------------------------------------------------------------
// Project 2.33 : Structure-Based Pharmacophore Modeling from MD Ensembles
//
// ROLE IN THE PROJECT
//   This is the "ground truth" the GPU result is checked against. It is written
//   to be OBVIOUSLY correct -- one readable loop over molecules, calling the SAME
//   score_molecule() (from pharmacophore.h) the GPU kernel calls -- so that when
//   the GPU and CPU agree, we believe the GPU.
//
//   Compiled by the host C++ compiler only (no CUDA here). See reference_cpu.h.
//
// READ THIS AFTER: reference_cpu.h, pharmacophore.h. Compare against kernels.cu.
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>     // std::ifstream
#include <stdexcept>   // std::runtime_error

// ---------------------------------------------------------------------------
// read_feature: pull one "type x y z weight" record out of the stream into a
//   Feature. Factored out because the loader reads features in two places (the
//   query block and each library molecule's block). Validates the type so a
//   garbled file cannot smuggle in an out-of-range enum that would later index
//   nothing meaningful.
// ---------------------------------------------------------------------------
static Feature read_feature(std::istream& in, const std::string& path) {
    Feature f{};
    if (!(in >> f.type >> f.x >> f.y >> f.z >> f.weight))
        throw std::runtime_error("feature row truncated in " + path);
    if (f.type < 0 || f.type >= FEAT_NUM_TYPES)
        throw std::runtime_error("feature type out of range in " + path);
    f._pad = 0;   // keep the padding deterministic (matters for byte-for-byte copies)
    return f;
}

ScreenData load_screen(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open screen file: " + path);

    ScreenData s;
    int n_query = 0;
    // Header: N library molecules, n_query query features, target index.
    if (!(in >> s.N >> n_query >> s.target) || s.N <= 0 || n_query <= 0)
        throw std::runtime_error("bad header (expected 'N n_query target') in " + path);

    // ---- query pharmacophore block ----
    s.query.reserve(n_query);
    for (int i = 0; i < n_query; ++i)
        s.query.push_back(read_feature(in, path));

    // ---- library molecules, into the flat CSR layout ----
    // offset[0] = 0; offset[k+1] = offset[k] + (#features on molecule k).
    s.offset.assign(s.N + 1, 0);
    for (int k = 0; k < s.N; ++k) {
        int m = 0;   // number of features on molecule k
        if (!(in >> m) || m < 0)
            throw std::runtime_error("bad feature count for a library molecule in " + path);
        for (int j = 0; j < m; ++j)
            s.lib_feats.push_back(read_feature(in, path));
        s.offset[k + 1] = s.offset[k] + m;   // running prefix sum -> CSR offsets
    }
    return s;
}

double query_self_overlap(const ScreenData& s) {
    // O_qq = sum over ALL ordered query-feature pairs of overlap_pair(i,j).
    // Summing all ordered pairs (both i,j and j,i, and i==j) matches exactly how
    // score_molecule() computes the LIBRARY self-overlap O_ll, so the Tanimoto
    // denominator (O_qq + O_ll - O_ql) is internally consistent.
    double o_qq = 0.0;
    const int nq = static_cast<int>(s.query.size());
    for (int i = 0; i < nq; ++i)
        for (int j = 0; j < nq; ++j)
            o_qq += overlap_pair(s.query[i], s.query[j]);
    return o_qq;
}

void screen_cpu(const ScreenData& s, double self_qq, std::vector<float>& scores) {
    scores.assign(static_cast<std::size_t>(s.N), 0.0f);
    const Feature* q = s.query.data();
    const int nq = static_cast<int>(s.query.size());

    // One molecule at a time. Each score is independent of the others -- which is
    // precisely why kernels.cu can hand each molecule to its own GPU thread.
    for (int k = 0; k < s.N; ++k) {
        const int beg = s.offset[k];          // first feature of molecule k
        const int n_k = s.offset[k + 1] - beg;  // feature count of molecule k
        const Feature* lib_k = s.lib_feats.data() + beg;
        // The ONE TRUE formula, shared with the GPU (pharmacophore.h).
        scores[static_cast<std::size_t>(k)] = score_molecule(q, nq, self_qq, lib_k, n_k);
    }
}
