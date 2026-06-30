// ===========================================================================
// src/kernels.cu  --  GPU twin of the ICE row-sum reduction (+ host driver)
// ---------------------------------------------------------------------------
// Project 3.15 : Hi-C / 3D Genome Contact Analysis
//
// This file implements the GPU hot loop of ICE matrix balancing: the per-bin
// row sum of the bias-corrected sparse contact matrix, computed with one thread
// per nonzero and a deterministic fixed-point atomic reduction (hic.h). The
// cheap O(n) bias update runs on the host via the SHARED ice_update_bias() that
// the CPU reference also calls, so ice_balance_gpu() and ice_balance_cpu()
// produce bit-identical biases. main.cu verifies exactly that.
//
// Why fixed-point integers (recap of hic.h): float atomicAdd sums in a
// nondeterministic order and float addition is non-associative, so a float tally
// would vary run-to-run and would not match the serial CPU sum. Quantizing to
// 64-bit integer quanta makes the colliding adds commute -> deterministic AND
// exactly CPU-matching. See docs/PATTERNS.md §3.
//
// READ THIS AFTER: kernels.cuh, hic.h, reference_cpu.h.
// ===========================================================================
#include "kernels.cuh"
#include "hic.h"               // hic_corrected, hic_to_fixed, hic_from_fixed
#include "util/cuda_check.cuh" // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"      // GpuTimer

#include <vector>

// Threads per block. 256 is a solid occupancy default across sm_75..sm_89: a
// multiple of the 32-lane warp, and small enough that the ragged last block
// wastes few lanes.
static constexpr int THREADS_PER_BLOCK = 256;

// ---------------------------------------------------------------------------
// ice_rowsum_kernel: each thread handles ONE nonzero and scatters its
//   fixed-point balanced value into the row-sum accumulators of its endpoints.
//
//   Thread/data mapping: t = blockIdx.x*blockDim.x + threadIdx.x owns nonzero t.
//   Guard `t >= nnz` retires the extra lanes of the last (ragged) block.
//
//   Memory: reads ei/ej/ecount/bias from GLOBAL memory; writes via atomicAdd to
//   the GLOBAL accumulator `acc`. Many nonzeros share a row, so their adds into
//   acc[i] collide -> we MUST use an atomic. atomicAdd on unsigned long long is
//   a single hardware instruction (supported since sm_35; our floor is sm_75).
//
//   Symmetry: an off-diagonal entry (i<j) exists at (i,j) and (j,i) in the full
//   matrix, so it feeds BOTH row i and row j. A diagonal entry feeds row i once.
//   This is identical to compute_rowsums_cpu() -> the two tallies match exactly.
// ---------------------------------------------------------------------------
__global__ void ice_rowsum_kernel(const int* __restrict__ ei,
                                  const int* __restrict__ ej,
                                  const double* __restrict__ ecount,
                                  long long nnz,
                                  const double* __restrict__ bias,
                                  unsigned long long* __restrict__ acc) {
    const long long t = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= nnz) return;   // retire lanes past the end of the entry array

    const int    i = ei[t];
    const int    j = ej[t];
    const double c = ecount[t];

    // Balanced contact M'_{ij} = count / (b_i b_j), quantized to integer quanta.
    // hic_corrected returns 0 if either endpoint is masked (bias 0) -> no effect.
    const double corrected = hic_corrected(c, bias[i], bias[j]);
    const unsigned long long q = hic_to_fixed(corrected);

    atomicAdd(&acc[i], q);                 // this nonzero contributes to row i
    if (i != j) atomicAdd(&acc[j], q);     // off-diagonal also contributes to row j
}

// ---------------------------------------------------------------------------
// ice_balance_gpu: the host driver. Mirrors ice_balance_cpu() but offloads the
//   row-sum reduction to the GPU. The bias is initialised on the host (occupied
//   bins -> 1, empty bins -> 0) and updated on the host each iteration with the
//   SHARED ice_update_bias(), so only the reduction differs from the CPU path.
//
//   Per iteration: upload current bias, zero the device accumulators, launch the
//   kernel, copy the fixed-point sums back, convert to doubles, run the host
//   update. We time only the kernel (CUDA events) and accumulate over iterations.
// ---------------------------------------------------------------------------
double ice_balance_gpu(const HicMatrix& m, int iters,
                       std::vector<double>& bias, float* kernel_ms) {
    const int n = m.n;
    const long long nnz = static_cast<long long>(m.entries.size());

    // -- Repack COO into struct-of-arrays so device loads coalesce. The CooEntry
    //    struct is array-of-structs (good for the CPU); on the GPU, separate i/j/
    //    count arrays let consecutive threads read consecutive addresses. --
    std::vector<int>    h_i(static_cast<std::size_t>(nnz));
    std::vector<int>    h_j(static_cast<std::size_t>(nnz));
    std::vector<double> h_c(static_cast<std::size_t>(nnz));
    for (long long t = 0; t < nnz; ++t) {
        h_i[t] = m.entries[t].i;
        h_j[t] = m.entries[t].j;
        h_c[t] = m.entries[t].count;
    }

    // -- Initialise bias on the host: occupied bins start at 1, empty at 0. This
    //    is byte-identical to ice_balance_cpu()'s initialisation. --
    std::vector<char> occupied(static_cast<std::size_t>(n), 0);
    for (const CooEntry& e : m.entries) { occupied[e.i] = 1; occupied[e.j] = 1; }
    bias.assign(static_cast<std::size_t>(n), 0.0);
    for (int k = 0; k < n; ++k) bias[k] = occupied[k] ? 1.0 : 0.0;

    // -- Device buffers. ei/ej/ecount are uploaded ONCE (the matrix is constant);
    //    bias is re-uploaded each iteration (it changes); acc is the tally. --
    int*    d_i = nullptr; int* d_j = nullptr; double* d_c = nullptr;
    double* d_bias = nullptr;
    unsigned long long* d_acc = nullptr;
    CUDA_CHECK(cudaMalloc(&d_i, static_cast<std::size_t>(nnz) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_j, static_cast<std::size_t>(nnz) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_c, static_cast<std::size_t>(nnz) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_bias, static_cast<std::size_t>(n) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_acc, static_cast<std::size_t>(n) * sizeof(unsigned long long)));

    CUDA_CHECK(cudaMemcpy(d_i, h_i.data(), static_cast<std::size_t>(nnz) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_j, h_j.data(), static_cast<std::size_t>(nnz) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c, h_c.data(), static_cast<std::size_t>(nnz) * sizeof(double),
                          cudaMemcpyHostToDevice));

    std::vector<unsigned long long> h_acc(static_cast<std::size_t>(n));
    std::vector<double> rowsum(static_cast<std::size_t>(n));
    const int grid = static_cast<int>((nnz + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    GpuTimer timer;
    float total_ms = 0.0f;
    double var = 0.0;

    for (int it = 0; it < iters; ++it) {
        // Upload the current bias and clear the integer accumulators.
        CUDA_CHECK(cudaMemcpy(d_bias, bias.data(), static_cast<std::size_t>(n) * sizeof(double),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_acc, 0, static_cast<std::size_t>(n) * sizeof(unsigned long long)));

        // The reduction (timed): one thread per nonzero, fixed-point atomicAdd.
        timer.start();
        ice_rowsum_kernel<<<grid, THREADS_PER_BLOCK>>>(d_i, d_j, d_c, nnz, d_bias, d_acc);
        total_ms += timer.stop_ms();
        CUDA_CHECK_LAST("ice_rowsum_kernel");

        // Bring the fixed-point sums back and convert to doubles (same as CPU).
        CUDA_CHECK(cudaMemcpy(h_acc.data(), d_acc,
                              static_cast<std::size_t>(n) * sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost));
        for (int k = 0; k < n; ++k) rowsum[k] = hic_from_fixed(h_acc[k]);

        // Host bias update -- the SAME helper the CPU reference calls.
        var = ice_update_bias(rowsum, bias);
    }
    *kernel_ms = total_ms;

    CUDA_CHECK(cudaFree(d_i));
    CUDA_CHECK(cudaFree(d_j));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFree(d_acc));
    return var;
}
