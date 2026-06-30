// ===========================================================================
// src/kernels.cu  --  EM iteration kernel (E-step + atomic M-step) + host loop
// ---------------------------------------------------------------------------
// Project 3.22 : RNA-seq Quantification / Pseudo-alignment
//
// GPU twin of em_cpu(): the SAME per-ec E-step (pseudoalign.h) and the SAME
// fixed-point M-step accumulation, plus the SAME host renormalise (counts_to_rho)
// reused from reference_cpu.cpp. Because every arithmetic step is identical and
// the only "reduction" uses commuting integer atomics, the GPU's final
// abundances match the CPU's exactly. main.cu compares them. See ../THEORY.md
// "The GPU mapping".
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"

#include <vector>

// 128 threads/block: ecs are tiny (a handful of members), so the kernel is
// memory-latency bound on the gather of member rho values, not compute bound. A
// modest block size keeps occupancy high without wasting registers; 64/128/256
// all behave similarly here (an easy thing for the learner to sweep).
static constexpr int THREADS_PER_BLOCK = 128;

// ---------------------------------------------------------------------------
// em_iteration_kernel: one EM iteration's E-step + M-step for ONE ec per thread.
//
//   grid   : enough blocks to cover M equivalence classes
//   block  : THREADS_PER_BLOCK threads
//   thread (blockIdx.x, threadIdx.x) -> equivalence-class index e = bx*bd + tx
//
//   Each thread:
//     1. finds its ec's member slice in the CSR arrays (d_ec_offset/d_ec_members),
//     2. runs psa_ec_contributions() to split the ec's reads among members
//        (identical math to the CPU; results held in a small per-thread scratch
//        in registers/local memory -- sized by PSA_MAX_EC_SIZE),
//     3. atomic-adds each member's fixed-point expected reads into the shared
//        per-transcript accumulator d_fixed_counts.
//
//   MEMORY SPACES: d_rho/d_eff_len/d_ec_* are read-only global (marked
//   __restrict__ so the compiler may cache through the read-only path). The only
//   writes are atomicAdd into d_fixed_counts (global). No shared memory is needed
//   because the per-ec work is tiny and independent.
//
//   WHY INTEGER ATOMICS: many ecs scatter into the same transcript (popular
//   isoforms), so the adds COLLIDE -> we need atomicAdd. atomicAdd on
//   `unsigned long long` is supported on all our target arches (sm_75+). Integer
//   adds are associative/commutative, so the result is independent of thread
//   order -> reproducible AND bit-identical to the CPU (PATTERNS.md section 3).
// ---------------------------------------------------------------------------
__global__ void em_iteration_kernel(const double*       __restrict__ d_rho,
                                    const double*       __restrict__ d_eff_len,
                                    const double*       __restrict__ d_ec_count,
                                    const std::int32_t* __restrict__ d_ec_offset,
                                    const std::int32_t* __restrict__ d_ec_members,
                                    int M,
                                    unsigned long long* __restrict__ d_fixed_counts) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's ec
    if (e >= M) return;                                    // guard ragged last block

    const std::int32_t base = d_ec_offset[e];
    const int k = d_ec_offset[e + 1] - base;               // members in this ec

    // Per-thread scratch for the k expected counts. Lives in registers/local
    // memory; bounded by PSA_MAX_EC_SIZE so no dynamic allocation in the kernel.
    double contrib[PSA_MAX_EC_SIZE];

    // E-step: identical to the CPU path (shared __host__ __device__ function).
    psa_ec_contributions(d_ec_count[e], &d_ec_members[base], k,
                         d_rho, d_eff_len, contrib);

    // M-step scatter: atomic fixed-point add into each member's accumulator.
    for (int j = 0; j < k; ++j) {
        const std::int32_t t = d_ec_members[base + j];
        atomicAdd(&d_fixed_counts[t], psa_to_fixed(contrib[j]));
    }
}

// ---------------------------------------------------------------------------
// em_gpu: host wrapper that runs `iters` EM iterations on the GPU.
//
//   Per iteration we: upload the current rho, zero the fixed-point accumulator,
//   launch one thread per ec, copy the accumulator back, and finish the M-step on
//   the host with counts_to_rho() (the exact same renormalise the CPU uses). The
//   rho upload + counts copy each iteration are small (length T) and keep CPU and
//   GPU perfectly in lockstep; a fully-on-device version (keeping rho resident and
//   renormalising with a reduction kernel) is left as a THEORY.md exercise.
//
//   The static (membership) arrays are uploaded ONCE before the loop -- they do
//   not change between iterations -- which is the I/O we would overlap with
//   compute using CUDA streams in a production build (catalog "CUDA streams for
//   I/O and compute overlap").
// ---------------------------------------------------------------------------
double em_gpu(const EcDataset& d, int iters,
              std::vector<double>& rho, std::vector<double>& est_counts,
              float* kernel_ms) {
    const int T = d.T, M = d.M;
    init_rho_uniform(d, rho);                              // same start as CPU

    // ---- Device buffers --------------------------------------------------
    double*       d_rho        = nullptr;   // [T] current abundances (uploaded each iter)
    double*       d_eff_len    = nullptr;   // [T] effective lengths (static)
    double*       d_ec_count   = nullptr;   // [M] reads per ec (static)
    std::int32_t* d_ec_offset  = nullptr;   // [M+1] CSR offsets (static)
    std::int32_t* d_ec_members = nullptr;   // [nnz] CSR member ids (static)
    unsigned long long* d_fixed_counts = nullptr;  // [T] M-step accumulator

    CUDA_CHECK(cudaMalloc(&d_rho,        static_cast<std::size_t>(T) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_eff_len,    static_cast<std::size_t>(T) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ec_count,   static_cast<std::size_t>(M) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ec_offset,  static_cast<std::size_t>(M + 1) * sizeof(std::int32_t)));
    CUDA_CHECK(cudaMalloc(&d_ec_members, d.ec_members.size() * sizeof(std::int32_t)));
    CUDA_CHECK(cudaMalloc(&d_fixed_counts, static_cast<std::size_t>(T) * sizeof(unsigned long long)));

    // Upload the STATIC arrays once (they never change across EM iterations).
    CUDA_CHECK(cudaMemcpy(d_eff_len, d.eff_len.data(),
                          static_cast<std::size_t>(T) * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ec_count, d.ec_count.data(),
                          static_cast<std::size_t>(M) * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ec_offset, d.ec_offset.data(),
                          static_cast<std::size_t>(M + 1) * sizeof(std::int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ec_members, d.ec_members.data(),
                          d.ec_members.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice));

    std::vector<unsigned long long> fixed_counts(T, 0ull);
    std::vector<double> prev_rho(T, 0.0);
    const int grid = (M + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    double last_delta = 0.0;

    GpuTimer timer;
    timer.start();
    for (int it = 0; it < iters; ++it) {
        prev_rho = rho;

        // Upload current rho; zero the accumulator; launch the E+M step.
        CUDA_CHECK(cudaMemcpy(d_rho, rho.data(),
                              static_cast<std::size_t>(T) * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_fixed_counts, 0,
                              static_cast<std::size_t>(T) * sizeof(unsigned long long)));
        em_iteration_kernel<<<grid, THREADS_PER_BLOCK>>>(
            d_rho, d_eff_len, d_ec_count, d_ec_offset, d_ec_members, M, d_fixed_counts);
        CUDA_CHECK_LAST("em_iteration_kernel");

        // Bring the accumulator back; finish the M-step on the host (same code
        // path as the CPU reference -> identical next rho).
        CUDA_CHECK(cudaMemcpy(fixed_counts.data(), d_fixed_counts,
                              static_cast<std::size_t>(T) * sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost));
        counts_to_rho(d, fixed_counts, rho);

        // Convergence witness (reported, not used to stop).
        last_delta = 0.0;
        for (int t = 0; t < T; ++t) last_delta += std::fabs(rho[t] - prev_rho[t]);
    }
    *kernel_ms = timer.stop_ms();

    // Final expected read counts from the last fixed-point sums.
    est_counts.assign(T, 0.0);
    for (int t = 0; t < T; ++t) est_counts[t] = psa_from_fixed(fixed_counts[t]);

    // ---- Free device memory ----------------------------------------------
    CUDA_CHECK(cudaFree(d_rho));
    CUDA_CHECK(cudaFree(d_eff_len));
    CUDA_CHECK(cudaFree(d_ec_count));
    CUDA_CHECK(cudaFree(d_ec_offset));
    CUDA_CHECK(cudaFree(d_ec_members));
    CUDA_CHECK(cudaFree(d_fixed_counts));
    return last_delta;
}
