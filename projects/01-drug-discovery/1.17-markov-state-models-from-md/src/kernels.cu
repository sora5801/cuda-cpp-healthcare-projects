// ===========================================================================
// src/kernels.cu  --  GPU MSM kernels (assign + accumulate + count) + driver
// ---------------------------------------------------------------------------
// Project 1.17 : Markov State Models from MD
//
// GPU twin of msm_cpu(): the SAME assign / fixed-point accumulate (msm.h) and
// the SAME integer transition count, then the SAME host transition-matrix and
// spectral helpers from reference_cpu.cpp. Because the parallel parts use only
// integer / fixed-point atomics (which commute), the GPU result is reproducible
// AND equals the CPU result exactly. main.cu compares the two.
//
// READ THIS AFTER: kernels.cuh (declarations + the thread-mapping idea), msm.h.
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// Threads per block. 256 is a solid default on sm_75..sm_89: a multiple of the
// 32-lane warp, enough warps to hide memory latency, many blocks resident.
static constexpr int THREADS_PER_BLOCK = 256;

// ---------------------------------------------------------------------------
// (A) assign_kernel: one thread per frame -> its nearest microstate.
//   Thread i reads its D-vector x[i*D ..] and calls the SHARED km_nearest()
//   (msm.h) -- the identical routine the CPU uses -- so the assignment is the
//   same on both. Reads global memory only; no atomics, no shared memory; the
//   centroids array is small and stays in the L2/constant-ish cache via reuse.
// ---------------------------------------------------------------------------
__global__ void assign_kernel(const float* __restrict__ x, int N, int D,
                              const float* __restrict__ centroids, int K,
                              int* __restrict__ labels) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's frame
    if (i >= N) return;                                    // guard ragged last block
    labels[i] = km_nearest(x + static_cast<std::size_t>(i) * D, centroids, K, D);
}

// ---------------------------------------------------------------------------
// (B) accumulate_kernel: scatter each frame's coordinates into its microstate's
//   running sum. Many frames share a microstate -> the adds collide -> we use
//   atomicAdd. Accumulating FIXED-POINT integers (km_to_fixed) makes the adds
//   COMMUTE, so the result is independent of thread order -> deterministic and
//   bit-identical to the CPU's plain += loop. (A float atomicAdd here would be
//   non-associative and irreproducible -- the lesson of PATTERNS.md section 3.)
// ---------------------------------------------------------------------------
__global__ void accumulate_kernel(const float* __restrict__ x, int N, int D,
                                  const int* __restrict__ labels,
                                  unsigned long long* __restrict__ sum,
                                  unsigned int* __restrict__ count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const int k = labels[i];                               // this frame's microstate
    const float* p = x + static_cast<std::size_t>(i) * D;
    for (int d = 0; d < D; ++d)
        atomicAdd(&sum[static_cast<std::size_t>(k) * D + d], km_to_fixed(p[d]));
    atomicAdd(&count[k], 1u);                              // one more frame in microstate k
}

// ---------------------------------------------------------------------------
// (C) count_transitions_kernel: one thread per time index t scatters one
//   (from -> to) transition into the K x K count matrix. Threads with valid t
//   are those in [0, N-lag); the rest return. atomicAdd on unsigned int again
//   commutes -> the GPU count matrix equals the CPU one exactly, frame for
//   frame. This is the maximum-likelihood sufficient statistic for the MSM.
//
//   THREAD-TO-DATA MAP: t = blockIdx.x*blockDim.x + threadIdx.x; it reads
//   labels[t] and labels[t+lag] (both already on the device from step A).
// ---------------------------------------------------------------------------
__global__ void count_transitions_kernel(const int* __restrict__ labels, int N, int K, int lag,
                                         unsigned int* __restrict__ counts) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t + lag >= N) return;                              // no transition starts here
    const int from = labels[t];
    const int to   = labels[t + lag];
    atomicAdd(&counts[static_cast<std::size_t>(from) * K + to], 1u);
}

// ---------------------------------------------------------------------------
// msm_gpu: host wrapper running the WHOLE MSM on the GPU. It mirrors msm_cpu()
//   step for step, swapping the two hot loops for kernels and reusing every
//   host helper (init/update centroids, build T, pi, timescale) so the results
//   match exactly. The reported time covers the per-iteration kernels plus the
//   final transition-count kernel (a teaching artifact, not a benchmark).
// ---------------------------------------------------------------------------
MsmResult msm_gpu(const Dataset& d, int iters, float* kernel_ms) {
    const int N = d.N, D = d.D, K = d.K;
    MsmResult r;
    init_centroids(d, r.centroids);            // host: identical deterministic seeding to CPU
    r.labels.assign(N, 0);
    r.sizes.assign(K, 0);

    // --- Device buffers -----------------------------------------------------
    float* d_x = nullptr; float* d_centroids = nullptr; int* d_labels = nullptr;
    unsigned long long* d_sum = nullptr; unsigned int* d_count = nullptr;
    unsigned int* d_trans = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, d.x.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_centroids, static_cast<std::size_t>(K) * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_labels, static_cast<std::size_t>(N) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sum, static_cast<std::size_t>(K) * D * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_count, static_cast<std::size_t>(K) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_trans, static_cast<std::size_t>(K) * K * sizeof(unsigned int)));
    // Frames never change during the run -> upload once.
    CUDA_CHECK(cudaMemcpy(d_x, d.x.data(), d.x.size() * sizeof(float), cudaMemcpyHostToDevice));

    std::vector<unsigned long long> sum(static_cast<std::size_t>(K) * D);
    const int grid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    GpuTimer timer;
    timer.start();
    // --- k-means: ASSIGN + ACCUMULATE per Lloyd iteration -------------------
    for (int it = 0; it < iters; ++it) {
        // Upload the current centroids; ASSIGN every frame to its nearest.
        CUDA_CHECK(cudaMemcpy(d_centroids, r.centroids.data(),
                              static_cast<std::size_t>(K) * D * sizeof(float), cudaMemcpyHostToDevice));
        assign_kernel<<<grid, THREADS_PER_BLOCK>>>(d_x, N, D, d_centroids, K, d_labels);

        // ACCUMULATE: zero the tallies, then atomic-add fixed-point coordinates.
        CUDA_CHECK(cudaMemset(d_sum, 0, static_cast<std::size_t>(K) * D * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(d_count, 0, static_cast<std::size_t>(K) * sizeof(unsigned int)));
        accumulate_kernel<<<grid, THREADS_PER_BLOCK>>>(d_x, N, D, d_labels, d_sum, d_count);

        // Bring tallies back; UPDATE centroids on the host (same code as CPU).
        CUDA_CHECK(cudaMemcpy(sum.data(), d_sum,
                              static_cast<std::size_t>(K) * D * sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(r.sizes.data(), d_count,
                              static_cast<std::size_t>(K) * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        update_centroids(d, sum, r.sizes, r.centroids);
    }

    // --- transition COUNT on the final labels (one kernel) ------------------
    CUDA_CHECK(cudaMemset(d_trans, 0, static_cast<std::size_t>(K) * K * sizeof(unsigned int)));
    count_transitions_kernel<<<grid, THREADS_PER_BLOCK>>>(d_labels, N, K, d.lag, d_trans);
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("msm kernels");

    // Pull labels + the integer count matrix back to the host.
    CUDA_CHECK(cudaMemcpy(r.labels.data(), d_labels,
                          static_cast<std::size_t>(N) * sizeof(int), cudaMemcpyDeviceToHost));
    r.counts.assign(static_cast<std::size_t>(K) * K, 0u);
    CUDA_CHECK(cudaMemcpy(r.counts.data(), d_trans,
                          static_cast<std::size_t>(K) * K * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_centroids));
    CUDA_CHECK(cudaFree(d_labels));
    CUDA_CHECK(cudaFree(d_sum));
    CUDA_CHECK(cudaFree(d_count));
    CUDA_CHECK(cudaFree(d_trans));

    // --- transition matrix + spectral analysis (host, reused from CPU) ------
    build_transition_matrix(K, r.counts, r.T);
    stationary_distribution(K, r.T, r.pi);
    r.timescale = slowest_timescale(K, d.lag, r.T, &r.lambda2);
    return r;
}
