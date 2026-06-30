// ===========================================================================
// src/kernels.cu  --  BQSR GPU kernels (atomic table build + recalibrate) + glue
// ---------------------------------------------------------------------------
// Project 3.25 : Base Quality Score Recalibration (BQSR)
//
// WHAT THIS FILE DOES
//   The GPU twin of reference_cpu.cpp. Two kernels, both "one thread per base":
//     * accumulate_kernel  -- classify each base (bqsr.h) and atomicAdd integer
//                             counts into its covariate bin (the scatter-reduction
//                             pattern; deterministic because integer adds commute).
//     * recalibrate_kernel -- read the finished table and write each base's new
//                             quality = empirical_q(bin).
//   bqsr_gpu() is the host glue: allocate, copy, launch both kernels (timed with
//   CUDA events), copy back. main.cu runs this and the CPU reference and asserts
//   the integer tables and recalibrated qualities are IDENTICAL.
//
// READ THIS AFTER: kernels.cuh (declarations + thread-mapping idea), bqsr.h.
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// Threads per block. 256 is a solid default on sm_75..sm_89: a multiple of the
// 32-lane warp, 8 warps to hide memory latency, plenty of resident blocks.
static constexpr int THREADS_PER_BLOCK = 256;

// ---------------------------------------------------------------------------
// accumulate_kernel: tally the covariate table.
//   Thread-to-data map: g = blockIdx.x*blockDim.x + threadIdx.x is the global
//   base index (read = g/read_len, cycle = g%read_len -- decoded inside
//   classify_base). The final block is ragged, so guard g >= total_bases.
//
//   classify_base() returns false for masked / skipped bases (known variant, 'N',
//   out-of-range Q) -> those threads simply do nothing. For a surviving base it
//   yields the covariate bin and whether the base mismatched the reference.
//
//   THE REDUCTION: many bases map to the same bin, so the writes COLLIDE. We use
//   atomicAdd on UNSIGNED INTEGER counters. Integer addition is associative and
//   commutative, so the final counts do not depend on the (nondeterministic)
//   order in which warps retire -> the table is reproducible run-to-run and
//   equals the CPU's table exactly. (Contrast a float atomicAdd, which would lose
//   determinism -- the lesson shared with flagships 5.01 and 11.09.)
// ---------------------------------------------------------------------------
__global__ void accumulate_kernel(int total_bases, int read_len,
                                  const char* __restrict__ ref, int ref_len,
                                  const char* __restrict__ bases,
                                  const int* __restrict__ quals,
                                  const int* __restrict__ read_pos,
                                  const unsigned char* __restrict__ known,
                                  unsigned int* __restrict__ d_obs,
                                  unsigned int* __restrict__ d_err) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= total_bases) return;                 // guard the ragged last block

    int bin = 0, is_err = 0;
    // Shared per-base decision (identical to the CPU). Skipped base -> return.
    if (!classify_base(g, read_len, ref, ref_len, bases, quals,
                       read_pos, known, &bin, &is_err))
        return;

    // Scatter-reduce: +1 observation always, +is_err errors. Atomic because many
    // threads target the same bin; integer => commutes => deterministic.
    atomicAdd(&d_obs[bin], 1u);
    atomicAdd(&d_err[bin], static_cast<unsigned int>(is_err));
}

// ---------------------------------------------------------------------------
// recalibrate_kernel: write each base's recalibrated quality.
//   One thread per base, no atomics (each thread owns one output element). For a
//   surviving base the new quality is empirical_q of its bin; a skipped base or a
//   no-evidence bin keeps the original reported quality. Identical logic to
//   recalibrate_cpu, so the outputs match exactly.
// ---------------------------------------------------------------------------
__global__ void recalibrate_kernel(int total_bases, int read_len,
                                   const char* __restrict__ ref, int ref_len,
                                   const char* __restrict__ bases,
                                   const int* __restrict__ quals,
                                   const int* __restrict__ read_pos,
                                   const unsigned char* __restrict__ known,
                                   const unsigned int* __restrict__ d_obs,
                                   const unsigned int* __restrict__ d_err,
                                   int* __restrict__ d_newq) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= total_bases) return;

    const int orig = quals[g];                    // reported quality (fallback)
    int bin = 0, is_err = 0;
    if (!classify_base(g, read_len, ref, ref_len, bases, quals,
                       read_pos, known, &bin, &is_err)) {
        d_newq[g] = orig;                          // skipped base: keep reported
        return;
    }
    const int qe = empirical_q(d_obs[bin], d_err[bin]);
    d_newq[g] = (qe < 0) ? orig : qe;              // no evidence -> keep reported
}

// ---------------------------------------------------------------------------
// bqsr_gpu: host wrapper -- the canonical CUDA steps, with two kernels.
//   (1) allocate device buffers for the read arrays, reference, mask, the two
//       integer counter tables, and the recalibrated-quality output.
//   (2) copy inputs host->device; zero the counter tables with cudaMemset.
//   (3) launch accumulate_kernel (build the table) then recalibrate_kernel
//       (apply it) -- both timed together with CUDA events.
//   (4) copy the table and the new qualities device->host.
//   (5) free everything.
// ---------------------------------------------------------------------------
void bqsr_gpu(const Dataset& d,
              std::vector<unsigned int>& obs,
              std::vector<unsigned int>& err,
              std::vector<int>& new_quals,
              float* kernel_ms) {
    const int total = d.total_bases();
    const int ref_len = static_cast<int>(d.reference.size());

    obs.assign(static_cast<std::size_t>(NUM_BINS), 0u);
    err.assign(static_cast<std::size_t>(NUM_BINS), 0u);
    new_quals.assign(static_cast<std::size_t>(total), 0);

    // (1) Device buffers (d_ prefix marks DEVICE pointers -- never deref on host).
    char*          d_ref   = nullptr;
    char*          d_bases = nullptr;
    int*           d_quals = nullptr;
    int*           d_pos   = nullptr;
    unsigned char* d_known = nullptr;
    unsigned int*  d_obs   = nullptr;
    unsigned int*  d_err   = nullptr;
    int*           d_newq  = nullptr;

    CUDA_CHECK(cudaMalloc(&d_ref,   static_cast<std::size_t>(ref_len) * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&d_bases, static_cast<std::size_t>(total)   * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&d_quals, static_cast<std::size_t>(total)   * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pos,   static_cast<std::size_t>(d.num_reads) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_known, static_cast<std::size_t>(ref_len) * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_obs,   static_cast<std::size_t>(NUM_BINS) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_err,   static_cast<std::size_t>(NUM_BINS) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_newq,  static_cast<std::size_t>(total)   * sizeof(int)));

    // (2) Copy inputs H2D; zero the two counter tables before accumulating.
    CUDA_CHECK(cudaMemcpy(d_ref,   d.reference.data(), static_cast<std::size_t>(ref_len) * sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bases, d.read_bases.data(), static_cast<std::size_t>(total)  * sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_quals, d.read_quals.data(), static_cast<std::size_t>(total)  * sizeof(int),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos,   d.read_pos.data(),   static_cast<std::size_t>(d.num_reads) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_known, d.known_site.data(), static_cast<std::size_t>(ref_len) * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_obs, 0, static_cast<std::size_t>(NUM_BINS) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_err, 0, static_cast<std::size_t>(NUM_BINS) * sizeof(unsigned int)));

    // (3) Launch both kernels (one thread per base), timed with CUDA events.
    const int blocks = (total + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    GpuTimer timer;
    timer.start();
    accumulate_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        total, d.read_len, d_ref, ref_len, d_bases, d_quals, d_pos, d_known, d_obs, d_err);
    recalibrate_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        total, d.read_len, d_ref, ref_len, d_bases, d_quals, d_pos, d_known, d_obs, d_err, d_newq);
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("bqsr kernels");

    // (4) Bring the table + recalibrated qualities back to host vectors.
    CUDA_CHECK(cudaMemcpy(obs.data(), d_obs, static_cast<std::size_t>(NUM_BINS) * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(err.data(), d_err, static_cast<std::size_t>(NUM_BINS) * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(new_quals.data(), d_newq, static_cast<std::size_t>(total) * sizeof(int), cudaMemcpyDeviceToHost));

    // (5) Free all device allocations (no GPU garbage collector exists).
    CUDA_CHECK(cudaFree(d_ref));
    CUDA_CHECK(cudaFree(d_bases));
    CUDA_CHECK(cudaFree(d_quals));
    CUDA_CHECK(cudaFree(d_pos));
    CUDA_CHECK(cudaFree(d_known));
    CUDA_CHECK(cudaFree(d_obs));
    CUDA_CHECK(cudaFree(d_err));
    CUDA_CHECK(cudaFree(d_newq));
}
