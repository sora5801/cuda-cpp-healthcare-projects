// ===========================================================================
// src/reference_cpu.h  --  Prototypes of the plain-C++ reference computation
// ---------------------------------------------------------------------------
// Project 2.17 : Allosteric Network Analysis
//
// WHY A SEPARATE HEADER
//   reference_cpu.cpp is compiled by the plain C++ compiler (cl.exe / g++) and
//   must NOT see any CUDA/__global__ syntax, so its prototypes cannot live in
//   kernels.cuh. Both main.cu and reference_cpu.cpp include THIS pure-C++ header
//   so they agree on the data structures and function signatures.
//
// WHAT THE REFERENCE COMPUTES  (the same thing the GPU does, but readably)
//   1. load_trajectory  : parse data/sample into a flat [T*N*3] float array.
//   2. residue_means    : per-residue average position over all frames.
//   3. dcc_matrix_cpu   : the full N*N Dynamical Cross-Correlation matrix.
//                         (this is the step the GPU parallelizes; see kernels.cu)
//   4. build_contacts   : the residue contact graph (which residues are neighbors).
//   5. shortest_paths   : Floyd-Warshall all-pairs communication distances on the
//                         -log|C| weighted contact graph, with path reconstruction.
//
//   The CPU reference exists for two reasons (CLAUDE.md section 5):
//     (a) it is the readable baseline that makes the GPU speed-up legible, and
//     (b) the demo runs BOTH the CPU and GPU DCC and asserts they agree exactly.
//
// READ THIS AFTER: dcc_core.h (the shared per-pair physics these functions use).
// ===========================================================================
#pragma once

#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Trajectory: a tiny container for the parsed molecular-dynamics sample.
//   coords is FRAME-MAJOR (see dcc_core.h coord_index):
//       coords[(t*N + i)*3 + c]  = component c of residue i in frame t
//   The trajectory is of N Cα ("alpha carbon") pseudo-atoms over T frames.
//   site_allo / site_active are the two residues of interest (the putative
//   allosteric site and the functional/active site) annotated in the sample
//   header, so the demo can report the communication path between them.
// ---------------------------------------------------------------------------
struct Trajectory {
    int N = 0;                     // number of residues (Cα atoms)
    int T = 0;                     // number of frames
    int site_allo = -1;            // residue index of the annotated allosteric site
    int site_active = -1;          // residue index of the annotated active site
    std::vector<float> coords;     // [T*N*3] flat positions, frame-major
};

// Parse the whitespace/structured sample file into a Trajectory.
//   Throws std::runtime_error if the file is missing or malformed so demos fail
//   loudly instead of silently analyzing empty data.
Trajectory load_trajectory(const std::string& path);

// Per-residue mean position over all frames: mean[i*3 + c].
//   Output `mean` is resized to N*3. Computed in double for a stable average;
//   both CPU and GPU center the trajectory with these identical means.
void residue_means(const Trajectory& traj, std::vector<double>& mean);

// The full N*N DCC matrix on the CPU, row-major: C[i*N + j].
//   Uses dcc_pair() from dcc_core.h for every entry, so it is the exact same
//   math the GPU kernel runs. Output `C` is resized to N*N (stored as float to
//   match the GPU's returned matrix for a bit-exact comparison).
void dcc_matrix_cpu(const Trajectory& traj, const std::vector<double>& mean,
                    std::vector<float>& C);

// Build the residue contact graph from the mean (equilibrium) structure.
//   Two residues are "in contact" (an edge may exist) if their mean Cα-Cα
//   distance is <= cutoff angstroms, OR they are sequential neighbors (|i-j|==1,
//   always bonded along the backbone). Output `adj` is an N*N row-major boolean
//   (stored as char): adj[i*N + j] == 1 iff i and j are in contact.
void build_contacts(const Trajectory& traj, const std::vector<double>& mean,
                    double cutoff, std::vector<char>& adj);

// Floyd-Warshall all-pairs shortest paths on the weighted contact graph.
//   Edge weight between contacting residues i,j is comm_weight(C[i][j]) =
//   -log|C[i][j]| (dcc_core.h). Outputs:
//     dist[i*N + j] : shortest communication distance i -> j (INF if unreachable)
//     next[i*N + j] : successor of i on the shortest path to j (-1 if none),
//                     used to reconstruct the actual residue pathway.
//   Floyd-Warshall is O(N^3) but trivially correct and deterministic, which is
//   what we want for a teaching reference. INF is a large sentinel (1e30).
void shortest_paths(const std::vector<float>& C, const std::vector<char>& adj,
                    int N, std::vector<double>& dist, std::vector<int>& next);

// Reconstruct the residue pathway src -> dst from a Floyd-Warshall `next` table.
//   Returns the ordered list of residue indices [src, ..., dst]; empty if dst is
//   unreachable from src. Pure bookkeeping, identical on any machine.
std::vector<int> reconstruct_path(const std::vector<int>& next, int N,
                                  int src, int dst);
