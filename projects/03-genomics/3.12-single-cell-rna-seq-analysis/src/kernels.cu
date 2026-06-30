// ===========================================================================
// src/kernels.cu  --  GPU kernels: normalize + brute-force KNN graph
// ---------------------------------------------------------------------------
// Project 3.12 : Single-Cell RNA-seq Analysis  (reduced-scope teaching version)
//
// WHAT THIS FILE DOES
//   Implements the two device kernels (normalize_kernel, knn_kernel) and the
//   host wrapper (run_gpu) that allocates GPU memory, moves data, launches the
//   kernels, times them, and brings the result back. These kernels are the GPU
//   twin of the CPU reference in reference_cpu.cpp -- they call the SAME shared
//   inline math (scrna.h), so main.cu can verify them index-for-index.
//
//   Comment density target here is >= 1:1 (CLAUDE.md section 6.2): kernels are
//   the heart of the repo, so the reasoning behind every launch is spelled out.
//
// READ THIS AFTER: kernels.cuh (declarations) and scrna.h (the per-element math).
// ===========================================================================
#include "kernels.cuh"
#include "scrna.h"               // sc_normalize_entry, sc_knn_one_cell (shared math)
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

#include <cmath>                 // sqrtf
#include <cstddef>               // std::size_t

// Threads per block. 256 is a solid default on sm_75..sm_89: it is a multiple of
// the 32-lane warp, gives the scheduler 8 warps to hide memory latency, and
// leaves plenty of blocks resident for occupancy. Both kernels use it.
static constexpr int THREADS_PER_BLOCK = 256;

// ---------------------------------------------------------------------------
// normalize_kernel: ONE THREAD PER CELL normalizes that cell's whole row.
//   Launch config (set in run_gpu):
//     grid  = ceil(N / THREADS_PER_BLOCK) blocks
//     block = THREADS_PER_BLOCK threads
//   Thread-to-data map: cell index c = blockIdx.x * blockDim.x + threadIdx.x.
//   Memory: thread c reads its G raw counts from global memory and writes its G
//   normalized values back. Threads never touch each other's rows, so there is
//   NO communication, NO shared memory, NO atomics -- a clean parallel map.
//
//   Why per-cell and not per-entry: the library size (the per-cell total) is a
//   reduction over a cell's G genes. Doing the whole cell in one thread keeps
//   that reduction local (a private register accumulator) and matches the CPU's
//   summation order exactly, so the normalized values are bit-identical.
// ---------------------------------------------------------------------------
__global__ void normalize_kernel(const float* __restrict__ counts, int N, int G,
                                 double target_sum, float* __restrict__ normalized) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's cell
    if (c >= N) return;                                    // guard the ragged last block

    const float* row = counts + static_cast<std::size_t>(c) * G;   // cell c's raw counts

    // Library size = total counts in this cell (a private per-thread reduction;
    // SAME left-to-right order as the CPU, so the sum is bit-identical).
    double cell_total = 0.0;
    for (int g = 0; g < G; ++g) cell_total += static_cast<double>(row[g]);

    // Normalize each entry with the shared formula (counts-per-target + log1p).
    for (int g = 0; g < G; ++g) {
        const double val = sc_normalize_entry(static_cast<double>(row[g]),
                                              cell_total, target_sum);
        normalized[static_cast<std::size_t>(c) * G + g] = static_cast<float>(val);
    }
}

// ---------------------------------------------------------------------------
// knn_kernel: ONE THREAD PER QUERY CELL finds that cell's k nearest neighbours.
//   Launch config (set in run_gpu):
//     grid  = ceil(N / THREADS_PER_BLOCK) blocks
//     block = THREADS_PER_BLOCK threads
//   Thread-to-data map: query cell q = blockIdx.x * blockDim.x + threadIdx.x.
//   Memory: each thread reads its own query row plus ALL N rows of the
//   normalized matrix from global memory (this is the O(N^2) read traffic the
//   deep dive flags). It keeps a private length-k top list in registers/local
//   memory -- no shared memory or atomics, because each query's neighbour list
//   is independent of every other query's.
//
//   The actual scan + top-k maintenance is sc_knn_one_cell (scrna.h), the SAME
//   function the CPU loops. Candidate cells are scanned in increasing index
//   order with a strict-< tie-break, so the GPU and CPU produce IDENTICAL
//   neighbour lists -- including the order of equidistant ties. That is what
//   lets main.cu verify the integer indices exactly (tolerance 0).
//
//   Teaching note on the real world: production tools do NOT do this O(N^2)
//   brute force. They first reduce G genes to ~50 PCA components, then use an
//   APPROXIMATE nearest-neighbour index (Faiss/HNSW) to get O(N log N). We do
//   the exact version because it is verifiable; THEORY.md explains the upgrade.
// ---------------------------------------------------------------------------
__global__ void knn_kernel(const float* __restrict__ normalized, int N, int G, int k,
                           int* __restrict__ nbr_idx, float* __restrict__ nbr_dist) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's query cell
    if (q >= N) return;                                    // guard the ragged last block

    // Private top-k scratch in local memory (sized to the compile-time max so no
    // dynamic allocation is needed inside the kernel).
    int    idx[SC_MAX_K];
    double sq [SC_MAX_K];

    // The whole per-query computation -- shared verbatim with the CPU reference.
    sc_knn_one_cell(normalized, N, G, q, k, idx, sq);

    // Publish: store neighbour indices and the true Euclidean distance (sqrt of
    // the squared distance the ranking used; sqrt is monotonic so order is kept).
    for (int j = 0; j < k; ++j) {
        nbr_idx [static_cast<std::size_t>(q) * k + j] = idx[j];
        nbr_dist[static_cast<std::size_t>(q) * k + j] = sqrtf(static_cast<float>(sq[j]));
    }
}

// ---------------------------------------------------------------------------
// run_gpu: host wrapper. The canonical CUDA steps, for the two-kernel pipeline:
//   (1) allocate device memory  (2) copy raw counts host->device
//   (3) launch normalize_kernel  (4) launch knn_kernel (reads the normalized
//       matrix already on the device -- no extra round trip)
//   (5) copy the normalized matrix + neighbour graph device->host
//   (6) free device memory
// We time steps (3)+(4) with CUDA events so the reported figure is the kernel
// cost, not the PCIe transfer cost (transfers are discussed in THEORY).
// ---------------------------------------------------------------------------
void run_gpu(const Dataset& d, KnnGraph& out, float* kernel_ms) {
    const int N = d.N, G = d.G, k = d.k;
    const std::size_t counts_n = static_cast<std::size_t>(N) * G;   // matrix elements
    const std::size_t graph_n  = static_cast<std::size_t>(N) * k;   // graph edges

    // Host output buffers sized up front; we fill them from the device at the end.
    out.normalized.assign(counts_n, 0.0f);
    out.nbr_idx.assign(graph_n, -1);
    out.nbr_dist.assign(graph_n, 0.0f);

    // (1) Device buffers. The d_ prefix marks DEVICE pointers (CLAUDE.md 12):
    //     dereferencing one on the host would crash, so the naming matters.
    float* d_counts     = nullptr;   // [N*G] raw counts (input)
    float* d_normalized = nullptr;   // [N*G] normalized matrix (intermediate + output)
    int*   d_nbr_idx    = nullptr;   // [N*k] neighbour indices (output)
    float* d_nbr_dist   = nullptr;   // [N*k] neighbour distances (output)
    CUDA_CHECK(cudaMalloc(&d_counts,     counts_n * sizeof(float)));   // can fail: OOM
    CUDA_CHECK(cudaMalloc(&d_normalized, counts_n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_nbr_idx,    graph_n  * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_nbr_dist,   graph_n  * sizeof(float)));

    // (2) Upload the raw count matrix.
    CUDA_CHECK(cudaMemcpy(d_counts, d.counts.data(), counts_n * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Launch geometry: one thread per cell covers both kernels (normalize is per
    // cell; KNN is per query cell -> same N threads). Ceiling division rounds up.
    const int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    GpuTimer timer;
    timer.start();
    // (3) Normalize the whole matrix on the device.
    normalize_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_counts, N, G, d.target_sum, d_normalized);
    CUDA_CHECK_LAST("normalize_kernel");
    // (4) Build the KNN graph from the normalized matrix already on the device.
    knn_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_normalized, N, G, k, d_nbr_idx, d_nbr_dist);
    *kernel_ms = timer.stop_ms();        // GPU-measured time for both kernels
    CUDA_CHECK_LAST("knn_kernel");       // catch launch + execution errors

    // (5) Bring the normalized matrix and the neighbour graph back to the host.
    CUDA_CHECK(cudaMemcpy(out.normalized.data(), d_normalized, counts_n * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.nbr_idx.data(), d_nbr_idx, graph_n * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.nbr_dist.data(), d_nbr_dist, graph_n * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // (6) Always free what we allocated (no GPU garbage collector exists).
    CUDA_CHECK(cudaFree(d_counts));
    CUDA_CHECK(cudaFree(d_normalized));
    CUDA_CHECK(cudaFree(d_nbr_idx));
    CUDA_CHECK(cudaFree(d_nbr_dist));
}
