// ===========================================================================
// src/reference_cpu.h  --  Dataset + shared MSM helpers + CPU reference
// ---------------------------------------------------------------------------
// Project 1.17 : Markov State Models from MD
//
// Pure C++ (no CUDA constructs). The per-frame distance / fixed-point math is in
// msm.h. The helpers declared here are the parts of the MSM pipeline that are
// SHARED by both the CPU reference and the GPU wrapper so the two produce
// identical microstate assignments, count matrices, and transition matrices:
//   * init_centroids   -- deterministic k-means seeding (farthest-first).
//   * update_centroids -- divide fixed-point coordinate sums by counts.
//   * build_transition_matrix -- row-normalize a count matrix into T.
//   * stationary_distribution / slowest_timescale -- the tiny host eigen-step.
// The GPU only replaces the two HOT loops (assign + count); everything else is
// reused, which is exactly why CPU and GPU agree exactly.
//
// READ THIS AFTER: msm.h. Then kernels.cuh (the GPU twin) and main.cu (driver).
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "msm.h"   // km_nearest, km_sqdist, km_to_fixed, MSM_SCALE

// ---------------------------------------------------------------------------
// Dataset: a featurized MD trajectory ready for an MSM.
//   x is the [N*D] feature matrix (row-major; frame i occupies x[i*D .. i*D+D)),
//   normalized to [0,1]. K is the number of microstates to cluster into. lag is
//   the MSM lag time tau in FRAMES: transitions are counted between frame t and
//   frame t+lag. Because the sample is a SINGLE continuous trajectory, frames
//   are time-ordered and consecutive in x.
// ---------------------------------------------------------------------------
struct Dataset {
    int N = 0;            // number of frames (time steps)
    int D = 0;            // feature dimension per frame
    int K = 0;            // number of microstates (k-means clusters)
    int lag = 1;          // MSM lag time tau, in frames
    std::vector<float> x; // [N*D] feature matrix, row-major, values in [0,1]
};

// MsmResult: the full output of one MSM build, so CPU and GPU results can be
// compared field by field in main.cu.
struct MsmResult {
    std::vector<float>        centroids; // [K*D] microstate centers
    std::vector<int>          labels;    // [N]   microstate of each frame
    std::vector<unsigned int> sizes;     // [K]   frames per microstate
    std::vector<unsigned int> counts;    // [K*K] transition COUNT matrix C (integers!)
    std::vector<double>       T;         // [K*K] transition PROBABILITY matrix (row-normalized C)
    std::vector<double>       pi;        // [K]   stationary distribution (pi T = pi)
    double                    timescale = 0.0; // slowest implied timescale t_2 (in frames)
    double                    lambda2   = 0.0; // 2nd-largest eigenvalue of T (magnitude)
};

// Load from the text format (data/README.md): header "N D K lag" then N rows of
// D floats. Throws std::runtime_error on a malformed file so demos fail loudly.
Dataset load_dataset(const std::string& path);

// Deterministic farthest-first init (the greedy seed of k-means++): center 0 is
// frame 0, then each next center is the frame farthest from all chosen centers.
// For well-separated conformational basins this seeds one microstate per basin.
void init_centroids(const Dataset& d, std::vector<float>& centroids);

// Centroid UPDATE shared by CPU/GPU: centroid[k][j] = (sum[k][j]/MSM_SCALE)/cnt.
// Empty microstates keep their previous centroid (so K stays fixed).
void update_centroids(const Dataset& d, const std::vector<unsigned long long>& sum,
                      const std::vector<unsigned int>& count, std::vector<float>& centroids);

// Inertia = sum over frames of squared distance to the assigned centroid (the
// k-means objective; lower is tighter). Shared metric for an apples-to-apples
// CPU/GPU comparison.
double compute_inertia(const Dataset& d, const std::vector<float>& centroids,
                       const std::vector<int>& labels);

// Build the K x K transition PROBABILITY matrix T from the integer COUNT matrix
// C by row-normalizing: T[i][j] = C[i][j] / sum_j C[i][j]. A row with zero
// outgoing transitions (a microstate never seen as a "from") becomes a self-loop
// (T[i][i]=1) so T stays a valid stochastic matrix. Shared by CPU/GPU.
void build_transition_matrix(int K, const std::vector<unsigned int>& counts,
                             std::vector<double>& T);

// Stationary distribution pi of T (the equilibrium microstate populations),
// found by power iteration on T^T (pi is the left eigenvector of eigenvalue 1).
// Returns pi normalized to sum 1. Tiny K x K work -> run on the host for both.
void stationary_distribution(int K, const std::vector<double>& T, std::vector<double>& pi);

// Slowest implied timescale t_2 = -lag / ln(lambda_2), where lambda_2 is the
// second-largest eigenvalue magnitude of T (the slowest relaxation mode). We get
// lambda_2 by deflated power iteration. Writes lambda_2 into *lambda2_out.
double slowest_timescale(int K, int lag, const std::vector<double>& T, double* lambda2_out);

// Shared, deterministic transition COUNT from a label sequence: for each t in
// [0, N-lag) increment counts[labels[t]*K + labels[t+lag]]. Used by the CPU
// reference; the GPU computes the SAME matrix with one thread per (t) pair.
void count_transitions_cpu(const Dataset& d, const std::vector<int>& labels,
                           std::vector<unsigned int>& counts);

// CPU reference: run the WHOLE MSM pipeline (k-means for `iters` Lloyd steps,
// transition counting at lag tau, T, pi, timescale). The trusted baseline that
// the GPU result is verified against.
MsmResult msm_cpu(const Dataset& d, int iters);
