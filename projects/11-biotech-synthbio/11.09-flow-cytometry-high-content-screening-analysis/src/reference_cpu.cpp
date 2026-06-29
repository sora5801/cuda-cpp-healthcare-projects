// ===========================================================================
// src/reference_cpu.cpp  --  Loader, shared helpers, serial k-means reference
// ---------------------------------------------------------------------------
// Project 11.09 : Flow Cytometry & High-Content Screening Analysis
// Compiled by the host compiler only. Math lives in kmeans.h.
// ===========================================================================
#include "reference_cpu.h"

#include <algorithm>
#include <fstream>
#include <limits>
#include <stdexcept>

Dataset load_dataset(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open dataset file: " + path);
    Dataset d;
    if (!(in >> d.N >> d.D >> d.K) || d.N <= 0 || d.D <= 0 || d.K <= 0 || d.K > d.N)
        throw std::runtime_error("bad header (expected 'N D K') in " + path);
    d.x.resize(static_cast<std::size_t>(d.N) * d.D);
    for (std::size_t i = 0; i < d.x.size(); ++i)
        if (!(in >> d.x[i])) throw std::runtime_error("dataset truncated in " + path);
    return d;
}

void init_centroids(const Dataset& d, std::vector<float>& centroids) {
    // Deterministic FARTHEST-FIRST init (the greedy heart of k-means++): start
    // at event 0, then repeatedly pick the event farthest from all centers
    // chosen so far. For well-separated populations this seeds one centroid per
    // population, avoiding the poor local minima that naive init can fall into.
    centroids.resize(static_cast<std::size_t>(d.K) * d.D);
    auto copy_center = [&](int k, int idx) {
        for (int j = 0; j < d.D; ++j)
            centroids[static_cast<std::size_t>(k) * d.D + j] =
                d.x[static_cast<std::size_t>(idx) * d.D + j];
    };
    copy_center(0, 0);                               // first center = event 0

    std::vector<double> min_d(d.N, std::numeric_limits<double>::infinity());
    for (int k = 1; k < d.K; ++k) {
        // Update each event's distance to the nearest chosen center (the one
        // just added, k-1), then pick the event that is farthest from all of
        // them (ties -> lowest index, for determinism).
        const float* last = &centroids[static_cast<std::size_t>(k - 1) * d.D];
        int best = 0; double best_d = -1.0;
        for (int i = 0; i < d.N; ++i) {
            const double dd = km_sqdist(&d.x[static_cast<std::size_t>(i) * d.D], last, d.D);
            if (dd < min_d[i]) min_d[i] = dd;
            if (min_d[i] > best_d) { best_d = min_d[i]; best = i; }
        }
        copy_center(k, best);
    }
}

void update_centroids(const Dataset& d, const std::vector<unsigned long long>& sum,
                      const std::vector<unsigned int>& count, std::vector<float>& centroids) {
    for (int k = 0; k < d.K; ++k) {
        if (count[k] == 0) continue;                 // empty cluster: keep old centroid
        for (int j = 0; j < d.D; ++j) {
            const double mean = (static_cast<double>(sum[static_cast<std::size_t>(k) * d.D + j])
                                 / KM_SCALE) / count[k];
            centroids[static_cast<std::size_t>(k) * d.D + j] = static_cast<float>(mean);
        }
    }
}

double compute_inertia(const Dataset& d, const std::vector<float>& centroids,
                       const std::vector<int>& labels) {
    double inertia = 0.0;
    for (int i = 0; i < d.N; ++i)
        inertia += km_sqdist(&d.x[static_cast<std::size_t>(i) * d.D],
                             &centroids[static_cast<std::size_t>(labels[i]) * d.D], d.D);
    return inertia;
}

double kmeans_cpu(const Dataset& d, int iters, std::vector<float>& centroids,
                  std::vector<int>& labels, std::vector<unsigned int>& sizes) {
    init_centroids(d, centroids);
    labels.assign(d.N, 0);
    std::vector<unsigned long long> sum(static_cast<std::size_t>(d.K) * d.D);
    sizes.assign(d.K, 0);

    for (int it = 0; it < iters; ++it) {
        // ASSIGN: nearest centroid for every event.
        for (int i = 0; i < d.N; ++i)
            labels[i] = km_nearest(&d.x[static_cast<std::size_t>(i) * d.D], centroids.data(), d.K, d.D);

        // UPDATE: fixed-point accumulate, then divide (same as the GPU does).
        std::fill(sum.begin(), sum.end(), 0ull);
        std::fill(sizes.begin(), sizes.end(), 0u);
        for (int i = 0; i < d.N; ++i) {
            const int k = labels[i];
            for (int j = 0; j < d.D; ++j)
                sum[static_cast<std::size_t>(k) * d.D + j] +=
                    km_to_fixed(d.x[static_cast<std::size_t>(i) * d.D + j]);
            sizes[k] += 1;
        }
        update_centroids(d, sum, sizes, centroids);
    }
    return compute_inertia(d, centroids, labels);
}
