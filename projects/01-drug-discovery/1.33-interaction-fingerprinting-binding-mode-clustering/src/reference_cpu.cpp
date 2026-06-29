// ===========================================================================
// src/reference_cpu.cpp  --  Loader, STAGE A, shared helpers, serial reference
// ---------------------------------------------------------------------------
// Project 1.33 : Interaction Fingerprinting & Binding-Mode Clustering
//
// Compiled by the host C++ compiler ONLY (no CUDA). All per-element math is in
// ifp.h, so every function here also runs (bit-identically) inside the GPU
// kernels. This file is written to be OBVIOUSLY correct -- plain loops, no
// cleverness -- because it is the ground truth the GPU is judged against.
//
// READ THIS AFTER: ifp.h, reference_cpu.h. Compare against kernels.cu (the twin).
// ===========================================================================
#include "reference_cpu.h"

#include <algorithm>   // std::fill
#include <fstream>
#include <stdexcept>

// ---------------------------------------------------------------------------
// load_dataset : parse the text format documented in data/README.md.
//   Layout:
//     line 1 : "P K"                                  (poses, clusters)
//     next NUM_RESIDUES lines: "x y z can_hbond can_aromatic can_ionic"
//     next P lines           : "x y z has_donor has_aromatic has_charge true_mode"
//   The trailing true_mode column is for REPORTING only (synthetic ground truth);
//   it never influences clustering. Throws on any malformed/truncated input so
//   demos fail loudly instead of silently clustering garbage.
// ---------------------------------------------------------------------------
Dataset load_dataset(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open dataset file: " + path);

    Dataset d;
    if (!(in >> d.P >> d.K) || d.P <= 0 || d.K <= 0 || d.K > d.P)
        throw std::runtime_error("bad header (expected 'P K') in " + path);

    // --- residues: a fixed pocket of NUM_RESIDUES rows ---------------------
    d.residues.resize(NUM_RESIDUES);
    for (int r = 0; r < NUM_RESIDUES; ++r) {
        Residue& R = d.residues[r];
        if (!(in >> R.x >> R.y >> R.z >> R.can_hbond >> R.can_aromatic >> R.can_ionic))
            throw std::runtime_error("residue block truncated in " + path);
    }

    // --- poses: P rows, each with a ground-truth mode label ----------------
    d.poses.resize(d.P);
    d.true_mode.resize(d.P);
    for (int p = 0; p < d.P; ++p) {
        Pose& P = d.poses[p];
        if (!(in >> P.x >> P.y >> P.z >> P.has_donor >> P.has_aromatic >> P.has_charge
                 >> d.true_mode[p]))
            throw std::runtime_error("pose block truncated in " + path);
    }
    return d;
}

// ---------------------------------------------------------------------------
// build_ifps : STAGE A on the CPU.
//   For each pose, start from an all-zero fingerprint and OR in each residue's
//   interaction nibble at that residue's bit offset. Bit index of (residue r,
//   type t) is  r*NUM_ITYPES + t; word = idx/64, bit-in-word = idx%64.
//   Complexity: O(P * NUM_RESIDUES). The GPU kernel parallelizes the outer P.
// ---------------------------------------------------------------------------
void build_ifps(const Dataset& d, std::vector<uint64_t>& fps) {
    fps.assign(static_cast<std::size_t>(d.P) * FP_WORDS, 0ull);
    for (int p = 0; p < d.P; ++p) {
        uint64_t* row = &fps[static_cast<std::size_t>(p) * FP_WORDS];   // pose p's IFP
        const Pose& pose = d.poses[p];
        for (int r = 0; r < NUM_RESIDUES; ++r) {
            // The shared geometry->bits routine (ifp.h): identical on the GPU.
            const int nibble = ifp_residue_nibble(pose, d.residues[r]);
            // Scatter the nibble's set bits to their global bit positions.
            for (int t = 0; t < NUM_ITYPES; ++t) {
                if (nibble & (1 << t)) {
                    const int idx  = r * NUM_ITYPES + t;   // global bit index
                    row[idx >> 6] |= (1ull << (idx & 63)); // set bit idx
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// init_centroids : deterministic farthest-first seeding (greedy k-means++).
//   centroid 0 = fingerprint of pose 0; then repeatedly add the fingerprint
//   whose distance to its NEAREST already-chosen centroid is largest. For
//   well-separated binding modes this lands one seed in each mode. Ties resolve
//   to the lowest index (strict > update), so the result is reproducible and
//   matches whatever the GPU wrapper does (it calls this same host function).
// ---------------------------------------------------------------------------
void init_centroids(const std::vector<uint64_t>& fps, int P, int K,
                    std::vector<uint64_t>& centroids) {
    centroids.assign(static_cast<std::size_t>(K) * FP_WORDS, 0ull);

    auto copy_fp = [&](int dst, int src) {
        for (int w = 0; w < FP_WORDS; ++w)
            centroids[static_cast<std::size_t>(dst) * FP_WORDS + w] =
                fps[static_cast<std::size_t>(src) * FP_WORDS + w];
    };
    copy_fp(0, 0);   // first centroid = pose 0's fingerprint

    // min_d[i] = distance of pose i to the nearest centroid chosen so far.
    std::vector<double> min_d(P, 2.0);   // 2.0 > any Tanimoto distance (<=1)
    for (int k = 1; k < K; ++k) {
        const uint64_t* last = &centroids[static_cast<std::size_t>(k - 1) * FP_WORDS];
        int best = 0; double best_d = -1.0;
        for (int i = 0; i < P; ++i) {
            const double dd = ifp_tanimoto_distance(
                &fps[static_cast<std::size_t>(i) * FP_WORDS], last);
            if (dd < min_d[i]) min_d[i] = dd;       // update nearest-center dist
            if (min_d[i] > best_d) { best_d = min_d[i]; best = i; }  // farthest pose
        }
        copy_fp(k, best);
    }
}

// ---------------------------------------------------------------------------
// update_centroids : per-bit majority vote (integer -> deterministic).
//   centroid k bit b is set iff at least half of cluster k's members set bit b:
//   2*count >= size  (ties keep the bit). Empty clusters are left unchanged so a
//   transiently-empty cluster can be re-populated on the next ASSIGN.
// ---------------------------------------------------------------------------
void update_centroids(int K, const std::vector<unsigned int>& bit_counts,
                      const std::vector<unsigned int>& sizes,
                      std::vector<uint64_t>& centroids) {
    for (int k = 0; k < K; ++k) {
        const unsigned int n = sizes[k];
        if (n == 0) continue;                       // empty cluster: keep old
        uint64_t* c = &centroids[static_cast<std::size_t>(k) * FP_WORDS];
        for (int w = 0; w < FP_WORDS; ++w) c[w] = 0ull;   // rebuild from scratch
        for (int b = 0; b < IFP_BITS; ++b) {
            const unsigned int cnt = bit_counts[static_cast<std::size_t>(k) * IFP_BITS + b];
            if (2u * cnt >= n)                      // majority (ties -> set)
                c[b >> 6] |= (1ull << (b & 63));
        }
    }
}

// ---------------------------------------------------------------------------
// cluster_cost : the k-means objective in Tanimoto space (sum of member-to-
//   centroid distances). Lower is tighter. A single number to report + compare.
// ---------------------------------------------------------------------------
double cluster_cost(const std::vector<uint64_t>& fps, int P,
                    const std::vector<uint64_t>& centroids,
                    const std::vector<int>& labels) {
    double cost = 0.0;
    for (int i = 0; i < P; ++i)
        cost += ifp_tanimoto_distance(
            &fps[static_cast<std::size_t>(i) * FP_WORDS],
            &centroids[static_cast<std::size_t>(labels[i]) * FP_WORDS]);
    return cost;
}

// ---------------------------------------------------------------------------
// ifp_cluster_cpu : the serial baseline. Lloyd's algorithm with a fixed iteration
//   count (deterministic -- no convergence test that could differ from the GPU).
//   Each iteration: ASSIGN every pose to its nearest consensus centroid, tally
//   per-cluster per-bit counts, then majority-vote the centroids. Returns cost.
// ---------------------------------------------------------------------------
double ifp_cluster_cpu(const std::vector<uint64_t>& fps, int P, int K, int iters,
                       std::vector<uint64_t>& centroids, std::vector<int>& labels,
                       std::vector<unsigned int>& sizes) {
    init_centroids(fps, P, K, centroids);
    labels.assign(P, 0);
    sizes.assign(K, 0);
    std::vector<unsigned int> bit_counts(static_cast<std::size_t>(K) * IFP_BITS);

    for (int it = 0; it < iters; ++it) {
        // ASSIGN: nearest consensus centroid for each pose (Tanimoto distance).
        for (int i = 0; i < P; ++i)
            labels[i] = ifp_nearest_centroid(
                &fps[static_cast<std::size_t>(i) * FP_WORDS], centroids.data(), K);

        // TALLY: per cluster, count how many members set each bit (integer).
        std::fill(bit_counts.begin(), bit_counts.end(), 0u);
        std::fill(sizes.begin(), sizes.end(), 0u);
        for (int i = 0; i < P; ++i) {
            const int k = labels[i];
            sizes[k] += 1u;
            const uint64_t* fp = &fps[static_cast<std::size_t>(i) * FP_WORDS];
            for (int b = 0; b < IFP_BITS; ++b)
                if (fp[b >> 6] & (1ull << (b & 63)))
                    bit_counts[static_cast<std::size_t>(k) * IFP_BITS + b] += 1u;
        }

        // UPDATE: majority-vote each centroid (shared with the GPU).
        update_centroids(K, bit_counts, sizes, centroids);
    }
    return cluster_cost(fps, P, centroids, labels);
}
