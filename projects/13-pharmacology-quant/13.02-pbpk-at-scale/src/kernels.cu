// ===========================================================================
// src/kernels.cu  --  PBPK population kernel (one thread per patient)
// ---------------------------------------------------------------------------
// Project 13.02 : PBPK at Scale
//
// GPU twin of integrate_cpu(): each thread runs the same RK4 loop (pbpk.h) for
// one virtual patient. main.cu compares the per-patient results. See ../THEORY.md.
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"

static constexpr int THREADS_PER_BLOCK = 128;

// Thread idx owns patient idx: sample physiology, integrate, summarize exposure.
// No inter-thread communication -- pure ensemble parallelism over the population.
__global__ void pbpk_kernel(PbpkParams P, PatientResult* __restrict__ results) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= P.n_patients) return;
    results[idx] = pbpk_integrate(P, idx);
}

void integrate_gpu(const PbpkParams& P, std::vector<PatientResult>& results, float* kernel_ms) {
    const int M = P.n_patients;
    results.assign(M, PatientResult{});

    PatientResult* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, static_cast<std::size_t>(M) * sizeof(PatientResult)));

    const int blocks = (M + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    GpuTimer timer;
    timer.start();
    pbpk_kernel<<<blocks, THREADS_PER_BLOCK>>>(P, d_out);
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("pbpk_kernel");

    CUDA_CHECK(cudaMemcpy(results.data(), d_out,
                          static_cast<std::size_t>(M) * sizeof(PatientResult),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_out));
}
