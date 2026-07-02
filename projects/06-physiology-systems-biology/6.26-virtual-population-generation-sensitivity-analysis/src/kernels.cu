// ===========================================================================
// src/kernels.cu  --  GPU Saltelli-evaluation kernel + host wrapper
// ---------------------------------------------------------------------------
// Project 6.26 : Virtual Population Generation & Sensitivity Analysis
//
// WHAT THIS FILE DOES
//   Implements the device kernel (evaluate_kernel) and the host glue
//   (evaluate_gpu) that allocates the device output, launches one thread per
//   Saltelli model evaluation, times the kernel with CUDA events, and copies the
//   AUC array back. This is the GPU twin of evaluate_cpu() in reference_cpu.cpp;
//   main.cu runs both, compares the raw arrays, then runs the Sobol reduction
//   (compute_sobol) on each and compares the resulting indices.
//
//   Every thread calls the SAME vpop_eval() used by the CPU reference (vpop.h),
//   so the GPU and CPU arrays agree to floating-point round-off. There are no
//   atomics and no shared memory here: the evaluations are fully independent
//   (embarrassingly parallel), which is exactly why Sobol-on-GPU is a big win
//   (see THEORY, "GPU mapping").
//
// READ THIS AFTER: kernels.cuh, vpop.h (the shared model + Saltelli sampling).
// ===========================================================================
#include "kernels.cuh"
#include "vpop.h"                // vpop_eval, vpop_num_evals (shared HD math)
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// Threads per block. 128 is a solid default on sm_75..sm_89 for a compute-bound
// per-thread workload (each thread runs a full trapezoid loop, so it is heavy on
// registers and math, light on memory): a multiple of the 32-lane warp, giving
// the scheduler 4 warps per block to hide the exp() latency while keeping many
// blocks resident. (128 vs 256 is a minor occupancy trade-off; either is fine.)
static constexpr int THREADS_PER_BLOCK = 128;

// ---------------------------------------------------------------------------
// evaluate_kernel: one thread computes one Saltelli model evaluation.
//   Launch config (set in evaluate_gpu):
//     grid  = ceil(total / THREADS_PER_BLOCK) blocks
//     block = THREADS_PER_BLOCK threads
//   Thread-to-data map: g = blockIdx.x * blockDim.x + threadIdx.x, where g is
//   the GLOBAL Saltelli index in [0, total). vpop_eval(P, g) decodes g into a
//   (block, row) Saltelli position, builds the parameter vector, and integrates
//   the PK model -- all in registers. Memory: writes exactly one double, out[g];
//   no reads from global memory except the tiny by-value P. No shared memory,
//   no atomics -- pure independent parallelism.
// ---------------------------------------------------------------------------
__global__ void evaluate_kernel(VpopParams P, long total, double* __restrict__ out) {
    // 64-bit global index: total = N*(k+2) can exceed 2^31 for large studies, so
    // we widen the thread index to long before comparing/indexing.
    const long g = (long)blockIdx.x * blockDim.x + threadIdx.x;

    // GUARD THE RAGGED LAST BLOCK: total is rarely an exact multiple of the block
    // size, so the final block has threads with g >= total. They must do nothing.
    if (g < total) {
        out[g] = vpop_eval(P, g);   // the SAME call the CPU reference makes
    }
}

// ---------------------------------------------------------------------------
// evaluate_gpu: host wrapper. The canonical CUDA steps, minus input copies (the
// only "input" is the tiny POD VpopParams, passed by value into the kernel):
//   (1) allocate the device output   (2) launch, timing the kernel with events
//   (3) copy the AUC array back       (4) free device memory
// We time ONLY the launch so the reported figure is the kernel cost, not the
// device->host transfer (discussed separately in THEORY).
// ---------------------------------------------------------------------------
void evaluate_gpu(const VpopParams& P, std::vector<double>& out, float* kernel_ms) {
    const long total = vpop_num_evals(P.N);                       // N*(k+2)
    out.assign(static_cast<std::size_t>(total), 0.0);
    const std::size_t bytes = static_cast<std::size_t>(total) * sizeof(double);

    // (1) Device output buffer. d_ prefix marks a DEVICE pointer (CLAUDE.md 12):
    //     dereferencing it on the host would crash, so the naming is load-bearing.
    double* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, bytes));   // can fail: out of device memory

    // (2) Launch. Blocks must cover all `total` evaluations, hence the ceiling
    //     division (total + B - 1) / B -- integer "round up". blocks is computed
    //     in 64-bit then narrowed: the grid X-dimension limit (2^31-1) is far
    //     above any teaching-scale study, so int is safe here.
    const int blocks = (int)((total + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    GpuTimer timer;
    timer.start();
    evaluate_kernel<<<blocks, THREADS_PER_BLOCK>>>(P, total, d_out);
    *kernel_ms = timer.stop_ms();            // GPU-measured kernel time
    CUDA_CHECK_LAST("evaluate_kernel");      // catch launch + execution errors

    // (3) Bring the AUC array back to the host vector.
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    // (4) Free the device buffer (no GPU garbage collector exists).
    CUDA_CHECK(cudaFree(d_out));
}
