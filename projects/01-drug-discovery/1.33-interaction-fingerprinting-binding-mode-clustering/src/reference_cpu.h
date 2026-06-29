// ===========================================================================
// src/reference_cpu.h  --  Dataset + shared helpers + CPU reference (no CUDA)
// ---------------------------------------------------------------------------
// Project 1.33 : Interaction Fingerprinting & Binding-Mode Clustering
//
// Pure C++ (compiled by cl.exe / g++, NEVER by nvcc). The per-element physics
// lives in ifp.h (geometry -> bits, Tanimoto distance, consensus). This header
// declares:
//   * Dataset            -- the loaded problem (residues, poses, K).
//   * build_ifps()       -- STAGE A on the CPU: poses -> packed IFP bit-vectors.
//   * the CONSENSUS-bit k-means helpers (init / update / objective), reused
//     verbatim by BOTH the CPU reference AND the GPU wrapper so the two produce
//     bit-identical clusters.
//   * ifp_cluster_cpu()  -- the trusted serial baseline the GPU is checked against.
//
// kernels.cu includes THIS header (for Dataset + the shared host helpers) and
// ifp.h (for the device-callable math). reference_cpu.cpp implements everything.
//
// READ THIS AFTER: ifp.h. Then read reference_cpu.cpp, then kernels.cuh.
// ===========================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "ifp.h"   // Residue, Pose, IFP_BITS, FP_WORDS, ifp_* math

// ---------------------------------------------------------------------------
// Dataset : everything one run needs.
//   The protein pocket is fixed (NUM_RESIDUES residues). The data file supplies
//   the residue geometry/chemistry and a list of P ligand poses, plus K, the
//   number of binding modes to cluster into.
// ---------------------------------------------------------------------------
struct Dataset {
    int P = 0;                       // number of ligand poses (rows to cluster)
    int K = 0;                       // number of binding-mode clusters to find
    std::vector<Residue> residues;   // [NUM_RESIDUES] pocket residues
    std::vector<Pose>    poses;      // [P] candidate ligand poses
    // The "true" mode label each pose was generated from (synthetic data only).
    // Used purely to REPORT how well clustering recovered the planted modes; it
    // is never read by the algorithm itself.
    std::vector<int>     true_mode;  // [P] ground-truth mode index, or empty
};

// Load the text format documented in data/README.md (header + residues + poses).
Dataset load_dataset(const std::string& path);

// ---------------------------------------------------------------------------
// STAGE A : build all P interaction fingerprints from the poses + residues.
//   Output `fps` is a flat [P * FP_WORDS] row-major array of 64-bit words: row p
//   is pose p's IFP. The CPU loops every (pose, residue) pair and ORs each
//   residue's nibble into the right place; the GPU kernel does the same in
//   parallel. Identical bit math (ifp.h) -> identical fingerprints.
// ---------------------------------------------------------------------------
void build_ifps(const Dataset& d, std::vector<uint64_t>& fps);

// ---------------------------------------------------------------------------
// STAGE B : consensus-bit k-means helpers (shared by CPU + GPU).
// ---------------------------------------------------------------------------
// Deterministic farthest-first init: centroid 0 = pose 0's fingerprint, then
// each next centroid = the fingerprint farthest (max Tanimoto distance) from all
// chosen so far. Ties -> lowest index. Seeds one centroid per distinct mode for
// well-separated data, dodging poor local minima (same idea as flagship 11.09).
void init_centroids(const std::vector<uint64_t>& fps, int P, int K,
                    std::vector<uint64_t>& centroids);

// UPDATE: rebuild each centroid as the per-bit MAJORITY vote of its members.
//   `bit_counts` is [K * IFP_BITS]: how many members of cluster k set bit b.
//   `sizes` is [K]: member count per cluster. Bit b of centroid k is set iff
//   2 * bit_counts >= sizes (strict majority, ties -> set), an integer test ->
//   order-independent and identical on CPU and GPU. Empty clusters keep their
//   old centroid. This is why the whole pipeline is BIT-EXACT, not just close.
void update_centroids(int K, const std::vector<unsigned int>& bit_counts,
                      const std::vector<unsigned int>& sizes,
                      std::vector<uint64_t>& centroids);

// OBJECTIVE: summed Tanimoto distance of every pose to its assigned centroid
//   (lower = tighter modes). Shared so CPU and GPU report the identical metric.
double cluster_cost(const std::vector<uint64_t>& fps, int P,
                    const std::vector<uint64_t>& centroids,
                    const std::vector<int>& labels);

// ---------------------------------------------------------------------------
// CPU REFERENCE: run `iters` Lloyd iterations of consensus-bit Tanimoto k-means.
//   Fills labels (P), centroids (K*FP_WORDS), sizes (K); returns the final cost.
//   The trusted baseline main.cu compares the GPU against.
// ---------------------------------------------------------------------------
double ifp_cluster_cpu(const std::vector<uint64_t>& fps, int P, int K, int iters,
                       std::vector<uint64_t>& centroids, std::vector<int>& labels,
                       std::vector<unsigned int>& sizes);
