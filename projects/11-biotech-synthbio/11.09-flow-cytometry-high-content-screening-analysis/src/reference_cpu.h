// ===========================================================================
// src/reference_cpu.h  --  Dataset + shared k-means helpers + CPU reference
// ---------------------------------------------------------------------------
// Project 11.09 : Flow Cytometry & High-Content Screening Analysis
//
// Pure C++ (no CUDA). The distance/fixed-point math is in kmeans.h. The centroid
// UPDATE (divide fixed-point sums by counts), the INERTIA, and the deterministic
// INIT are host helpers reused by BOTH the CPU reference and the GPU wrapper, so
// the two produce identical centroids. kernels.cu reuses Dataset + these helpers.
// ===========================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "kmeans.h"   // km_nearest, km_to_fixed, KM_SCALE

// A loaded dataset: N events, each D markers; cluster into K populations.
struct Dataset {
    int N = 0, D = 0, K = 0;
    std::vector<float> x;   // [N*D] events, row-major; normalized to [0,1]
};

// Load from the text format (data/README.md): "N D K" then N rows of D floats.
Dataset load_dataset(const std::string& path);

// Deterministic init: take K events at evenly spaced indices as the centroids.
void init_centroids(const Dataset& d, std::vector<float>& centroids);

// Update centroids from fixed-point coordinate sums + counts (shared by CPU/GPU):
//   centroid[k][d] = (sum[k][d] / KM_SCALE) / count[k]   (empty clusters: unchanged)
void update_centroids(const Dataset& d, const std::vector<unsigned long long>& sum,
                      const std::vector<unsigned int>& count, std::vector<float>& centroids);

// Inertia = sum over events of squared distance to the assigned centroid (the
// k-means objective; lower is tighter). Shared by CPU/GPU for an identical metric.
double compute_inertia(const Dataset& d, const std::vector<float>& centroids,
                       const std::vector<int>& labels);

// CPU reference: run `iters` Lloyd iterations. Fills labels (N), centroids (K*D),
// and cluster sizes (K); returns the final inertia. The trusted baseline.
double kmeans_cpu(const Dataset& d, int iters, std::vector<float>& centroids,
                  std::vector<int>& labels, std::vector<unsigned int>& sizes);
