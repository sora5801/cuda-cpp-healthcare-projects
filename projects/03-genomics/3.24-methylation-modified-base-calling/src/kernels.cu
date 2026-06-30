// ===========================================================================
// src/kernels.cu  --  Methylation-calling kernel + host wrapper
// ---------------------------------------------------------------------------
// Project 3.24 : Methylation / Modified-Base Calling
//
// GPU twin of score_jobs_cpu(): one thread per (read, site) job, the two pore
// models in constant memory, the per-job banded DP shared verbatim with the CPU
// (banded_align_core in meth_core.h). main.cu runs both and compares the LLRs.
// See ../THEORY.md "GPU mapping" for the full reasoning.
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"

#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// The two pore models in CONSTANT memory. Each is NUM_KMERS PoreModelEntry rows
// (64 * 2 floats = 512 B here; even a real k=9 model's 262144 rows is 2 MB, which
// would NOT fit constant memory -- THEORY.md "real world" notes that production
// f5c keeps big models in global/texture memory. At teaching size they fit
// comfortably, and constant memory is ideal: every thread reads the same rows, so
// the constant cache broadcasts each entry warp-wide instead of issuing 32 loads).
//
// Read by score_jobs_kernel via banded_align_core(); uploaded by the host wrapper
// with cudaMemcpyToSymbol before the launch.
// ---------------------------------------------------------------------------
__constant__ PoreModelEntry c_canon[NUM_KMERS];   // canonical (unmodified C) model
__constant__ PoreModelEntry c_meth [NUM_KMERS];   // methylated (5mC) model

// A block of 128 threads: a good occupancy default on sm_75..sm_89 for a kernel
// whose per-thread work is a small DP with a handful of double-precision FMAs.
static constexpr int THREADS_PER_BLOCK = 128;

// ---------------------------------------------------------------------------
// score_jobs_kernel: one thread computes one job's LLR.
//   grid   : ceil(num_jobs / THREADS_PER_BLOCK) blocks
//   block  : THREADS_PER_BLOCK threads
//   thread (blockIdx.x, threadIdx.x) -> job index j
//
//   Steps (identical to banded_align_logL on the CPU, then the difference):
//     1. Slide a KMER_K window over the job's WINDOW_BASES reference base codes to
//        build the WINDOW_KMERS k-mer indices (kmer_code from meth_core.h).
//     2. Run banded_align_core under the methylated model (constant memory).
//     3. Run it again under the canonical model.
//     4. Write LLR = logL_meth - logL_canon.
//   No atomics, no shared memory, no inter-thread communication -> embarrassingly
//   parallel and deterministic. The DP scratch (two rows of doubles) lives in
//   each thread's registers/local memory.
// ---------------------------------------------------------------------------
__global__ void score_jobs_kernel(const Job* __restrict__ jobs, int num_jobs,
                                  float* __restrict__ llr) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's job
    if (j >= num_jobs) return;                              // guard ragged last block

    const Job& job = jobs[j];

    // 1. Reference k-mer codes for this job's window (same as the CPU wrapper).
    int kmer_ids[WINDOW_KMERS];
    #pragma unroll
    for (int w = 0; w < WINDOW_KMERS; ++w)
        kmer_ids[w] = kmer_code(&job.ref_codes[w]);

    // 2 + 3. Score under both pore models using the SHARED DP core.
    const double logL_meth  = banded_align_core(job.events, kmer_ids, c_meth,
                                                EVENTS_PER_JOB, WINDOW_KMERS);
    const double logL_canon = banded_align_core(job.events, kmer_ids, c_canon,
                                                EVENTS_PER_JOB, WINDOW_KMERS);

    // 4. Log-likelihood ratio (methylated vs canonical). Cast to float to match
    //    the CPU reference's stored type exactly.
    llr[j] = (float)(logL_meth - logL_canon);
}

// ---------------------------------------------------------------------------
// score_jobs_gpu: orchestrate the whole GPU computation (see kernels.cuh).
// ---------------------------------------------------------------------------
void score_jobs_gpu(const MethData& d, std::vector<float>& llr, float* kernel_ms) {
    const int num_jobs = static_cast<int>(d.jobs.size());
    llr.assign(num_jobs, 0.0f);
    if (num_jobs == 0) { *kernel_ms = 0.0f; return; }

    // Upload both pore models to constant memory (cudaMemcpyToSymbol copies into
    // the named __constant__ array). They never change during the launch.
    CUDA_CHECK(cudaMemcpyToSymbol(c_canon, d.canon.data(), NUM_KMERS * sizeof(PoreModelEntry)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_meth,  d.meth.data(),  NUM_KMERS * sizeof(PoreModelEntry)));

    // Device buffers: the job array (input) and the LLR array (output).
    Job*   d_jobs = nullptr;
    float* d_llr  = nullptr;
    CUDA_CHECK(cudaMalloc(&d_jobs, static_cast<std::size_t>(num_jobs) * sizeof(Job)));
    CUDA_CHECK(cudaMalloc(&d_llr,  static_cast<std::size_t>(num_jobs) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_jobs, d.jobs.data(),
                          static_cast<std::size_t>(num_jobs) * sizeof(Job),
                          cudaMemcpyHostToDevice));

    const int grid = (num_jobs + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    GpuTimer timer;
    timer.start();
    score_jobs_kernel<<<grid, THREADS_PER_BLOCK>>>(d_jobs, num_jobs, d_llr);
    *kernel_ms = timer.stop_ms();        // blocks until the kernel finishes
    CUDA_CHECK_LAST("score_jobs_kernel");

    CUDA_CHECK(cudaMemcpy(llr.data(), d_llr,
                          static_cast<std::size_t>(num_jobs) * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_jobs));
    CUDA_CHECK(cudaFree(d_llr));
}
