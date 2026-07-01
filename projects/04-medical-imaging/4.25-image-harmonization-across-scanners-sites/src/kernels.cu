// ===========================================================================
// src/kernels.cu  --  GPU ComBat kernel (one thread per feature) + host wrapper
// ---------------------------------------------------------------------------
// Project 4.25 : Image Harmonization Across Scanners/Sites
//
// GPU twin of combat_cpu(): the SAME shared per-feature core cb_harmonize_feature
// (combat.h), invoked from one thread per feature instead of a serial host loop.
// Because both sides call the identical __host__ __device__ function on identical
// double-precision inputs, the harmonized tables match to ~machine precision.
// main.cu compares them. See ../THEORY.md "GPU mapping".
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

#include "combat.h"              // cb_harmonize_feature (shared core)

// A modest block width. ComBat's per-feature work is register-heavy (each thread
// keeps an M x M normal matrix + several length-M vectors in local memory, M <=
// CB_MAX_M=16), so we keep threads/block moderate to leave register headroom.
// 128 is a safe occupancy default on sm_75..sm_89; tune with Nsight if needed.
static constexpr int THREADS_PER_BLOCK = 128;

// ---------------------------------------------------------------------------
// combat_kernel: one thread harmonizes ONE feature row.
//   Launch config (set in combat_gpu):
//     grid  = ceil(P / THREADS_PER_BLOCK) blocks
//     block = THREADS_PER_BLOCK threads
//   Thread-to-data map: p = blockIdx.x * blockDim.x + threadIdx.x owns feature p,
//     i.e. rows d_Y[p*N .. p*N+N-1] (input) and d_out[p*N ..] (output).
//   Memory: reads its own feature row + the shared design/priors from GLOBAL
//     memory; all scratch (the normal matrix, beta, gamma/delta) lives in
//     REGISTERS/LOCAL memory inside cb_harmonize_feature. No shared memory, no
//     atomics -> features are fully independent, so the result is deterministic.
// ---------------------------------------------------------------------------
__global__ void combat_kernel(const double* __restrict__ d_Y,       // [P*N] features (row-major)
                              const double* __restrict__ d_design,  // [N*M] design (shared)
                              const int*    __restrict__ d_batch,   // [N] batch label per sample
                              int N, int P, int M, int Ccols, int B,
                              const int*    __restrict__ d_batch_n, // [B] samples per batch
                              const double* __restrict__ d_gamma_bar, // [B] EB priors
                              const double* __restrict__ d_tau2,      // [B]
                              const double* __restrict__ d_a_prior,   // [B]
                              const double* __restrict__ d_b_prior,   // [B]
                              double* __restrict__ d_out) {           // [P*N] harmonized
    const int p = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's feature
    // GUARD THE RAGGED LAST BLOCK: P is rarely a multiple of the block size, so
    // the final block has threads with p >= P; they must do nothing or they read
    // and write out of bounds (an illegal-address crash).
    if (p >= P) return;
    // Delegate the entire ComBat pipeline for this feature to the shared core.
    // Passing d_Y + p*N / d_out + p*N gives this thread its own input/output row.
    cb_harmonize_feature(
        d_Y + (std::size_t)p * N, d_design, d_batch,
        N, M, Ccols, B,
        d_batch_n, d_gamma_bar, d_tau2, d_a_prior, d_b_prior,
        d_out + (std::size_t)p * N);
}

// ---------------------------------------------------------------------------
// combat_gpu: host wrapper. The five canonical CUDA steps, each narrated:
//   (1) allocate device memory  (2) copy inputs host->device
//   (3) launch the kernel        (4) copy result device->host
//   (5) free device memory
// We time ONLY step (3) with CUDA events so the reported figure is kernel cost,
// not PCIe transfer cost (THEORY.md §GPU mapping discusses transfers separately).
// ---------------------------------------------------------------------------
void combat_gpu(const Dataset& d, const std::vector<double>& design,
                const std::vector<double>& gamma_bar, const std::vector<double>& tau2,
                const std::vector<double>& a_prior,   const std::vector<double>& b_prior,
                const std::vector<int>&    batch_n,
                std::vector<double>& out, float* kernel_ms) {
    const int N = d.N, P = d.P, M = d.M(), B = d.B, Ccols = d.Ccols();
    out.assign((std::size_t)P * N, 0.0);

    // ---- (1) Device pointers (d_ prefix = DEVICE memory, CLAUDE §12) ----------
    double *d_Y = nullptr, *d_design = nullptr, *d_out = nullptr;
    double *d_gamma_bar = nullptr, *d_tau2 = nullptr, *d_a_prior = nullptr, *d_b_prior = nullptr;
    int *d_batch = nullptr, *d_batch_n = nullptr;
    CUDA_CHECK(cudaMalloc(&d_Y,        d.Y.size()     * sizeof(double)));   // can fail: OOM
    CUDA_CHECK(cudaMalloc(&d_design,   design.size()  * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_out,      (std::size_t)P * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_batch,    (std::size_t)N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_batch_n,  (std::size_t)B * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_gamma_bar,(std::size_t)B * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_tau2,     (std::size_t)B * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_a_prior,  (std::size_t)B * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b_prior,  (std::size_t)B * sizeof(double)));

    // ---- (2) Upload the read-only inputs --------------------------------------
    CUDA_CHECK(cudaMemcpy(d_Y,       d.Y.data(),       d.Y.size()    * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_design,  design.data(),    design.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_batch,   d.batch.data(),   (std::size_t)N * sizeof(int),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_batch_n, batch_n.data(),   (std::size_t)B * sizeof(int),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma_bar, gamma_bar.data(),(std::size_t)B * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tau2,    tau2.data(),      (std::size_t)B * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_a_prior, a_prior.data(),   (std::size_t)B * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b_prior, b_prior.data(),   (std::size_t)B * sizeof(double), cudaMemcpyHostToDevice));

    // ---- (3) Launch: one thread per feature (timed) ---------------------------
    // Ceiling division so every feature is covered even when P is not a multiple
    // of the block size (the extra threads are stopped by the guard in the kernel).
    const int grid = (P + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    GpuTimer timer;
    timer.start();
    combat_kernel<<<grid, THREADS_PER_BLOCK>>>(
        d_Y, d_design, d_batch, N, P, M, Ccols, B,
        d_batch_n, d_gamma_bar, d_tau2, d_a_prior, d_b_prior, d_out);
    *kernel_ms = timer.stop_ms();     // blocks until the kernel finishes
    CUDA_CHECK_LAST("combat_kernel"); // catch launch-config and in-kernel errors

    // ---- (4) Download the harmonized table ------------------------------------
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, (std::size_t)P * N * sizeof(double), cudaMemcpyDeviceToHost));

    // ---- (5) Free device memory -----------------------------------------------
    CUDA_CHECK(cudaFree(d_Y));        CUDA_CHECK(cudaFree(d_design));   CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_batch));    CUDA_CHECK(cudaFree(d_batch_n));
    CUDA_CHECK(cudaFree(d_gamma_bar));CUDA_CHECK(cudaFree(d_tau2));
    CUDA_CHECK(cudaFree(d_a_prior));  CUDA_CHECK(cudaFree(d_b_prior));
}
