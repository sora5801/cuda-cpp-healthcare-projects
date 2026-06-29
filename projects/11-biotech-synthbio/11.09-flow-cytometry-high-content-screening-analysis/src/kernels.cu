// ===========================================================================
// src/kernels.cu  --  k-means kernels (assign + atomic accumulate) + loop
// ---------------------------------------------------------------------------
// Project 11.09 : Flow Cytometry & High-Content Screening Analysis
//
// GPU twin of kmeans_cpu(): identical assign + fixed-point accumulate (kmeans.h)
// and the SAME host centroid-update, so the results match exactly. main.cu
// compares labels + centroids. See ../THEORY.md "GPU mapping".
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"

static constexpr int THREADS_PER_BLOCK = 256;

__global__ void assign_kernel(const float* __restrict__ x, int N, int D,
                              const float* __restrict__ centroids, int K,
                              int* __restrict__ labels) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    labels[i] = km_nearest(x + static_cast<std::size_t>(i) * D, centroids, K, D);
}

// Each event scatters its fixed-point coordinates into its cluster's sum. Many
// events share a cluster, so the adds collide -> atomicAdd. Integer fixed-point
// makes the adds commute => order-independent => deterministic and CPU-matching.
__global__ void accumulate_kernel(const float* __restrict__ x, int N, int D,
                                  const int* __restrict__ labels,
                                  unsigned long long* __restrict__ sum,
                                  unsigned int* __restrict__ count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const int k = labels[i];
    const float* p = x + static_cast<std::size_t>(i) * D;
    for (int d = 0; d < D; ++d)
        atomicAdd(&sum[static_cast<std::size_t>(k) * D + d], km_to_fixed(p[d]));
    atomicAdd(&count[k], 1u);
}

double kmeans_gpu(const Dataset& d, int iters, std::vector<float>& centroids,
                  std::vector<int>& labels, std::vector<unsigned int>& sizes, float* kernel_ms) {
    const int N = d.N, D = d.D, K = d.K;
    init_centroids(d, centroids);                 // host: same deterministic init as CPU
    labels.assign(N, 0);
    sizes.assign(K, 0);

    // Device buffers.
    float* d_x = nullptr; float* d_centroids = nullptr; int* d_labels = nullptr;
    unsigned long long* d_sum = nullptr; unsigned int* d_count = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, d.x.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_centroids, static_cast<std::size_t>(K) * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_labels, static_cast<std::size_t>(N) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sum, static_cast<std::size_t>(K) * D * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_count, static_cast<std::size_t>(K) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(d_x, d.x.data(), d.x.size() * sizeof(float), cudaMemcpyHostToDevice));

    std::vector<unsigned long long> sum(static_cast<std::size_t>(K) * D);
    const int grid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    GpuTimer timer;
    timer.start();
    for (int it = 0; it < iters; ++it) {
        // Upload current centroids; ASSIGN every event to its nearest.
        CUDA_CHECK(cudaMemcpy(d_centroids, centroids.data(),
                              static_cast<std::size_t>(K) * D * sizeof(float), cudaMemcpyHostToDevice));
        assign_kernel<<<grid, THREADS_PER_BLOCK>>>(d_x, N, D, d_centroids, K, d_labels);

        // ACCUMULATE: zero the tallies, then atomic-add fixed-point coordinates.
        CUDA_CHECK(cudaMemset(d_sum, 0, static_cast<std::size_t>(K) * D * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(d_count, 0, static_cast<std::size_t>(K) * sizeof(unsigned int)));
        accumulate_kernel<<<grid, THREADS_PER_BLOCK>>>(d_x, N, D, d_labels, d_sum, d_count);

        // Bring the tallies back; UPDATE centroids on the host (same code as CPU).
        CUDA_CHECK(cudaMemcpy(sum.data(), d_sum,
                              static_cast<std::size_t>(K) * D * sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(sizes.data(), d_count,
                              static_cast<std::size_t>(K) * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        update_centroids(d, sum, sizes, centroids);
    }
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("kmeans kernels");

    CUDA_CHECK(cudaMemcpy(labels.data(), d_labels,
                          static_cast<std::size_t>(N) * sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_centroids));
    CUDA_CHECK(cudaFree(d_labels));
    CUDA_CHECK(cudaFree(d_sum));
    CUDA_CHECK(cudaFree(d_count));
    return compute_inertia(d, centroids, labels);
}
