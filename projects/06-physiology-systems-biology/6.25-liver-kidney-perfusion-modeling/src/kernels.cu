// ===========================================================================
// src/kernels.cu  --  Ensemble perfusion kernel (one thread per sinusoid)
// ---------------------------------------------------------------------------
// Project 6.25 : Liver & Kidney Perfusion Modeling
//
// WHAT THIS FILE DOES
//   The GPU twin of integrate_cpu(). Each thread runs the same RK4 spatial march
//   (perfusion.h) for one sinusoid's inlet velocity, then writes its
//   SinusoidResult. main.cu compares the per-sinusoid results against the CPU
//   reference. Because the physics/RK4 is SHARED (perfusion.h), the numbers match
//   to round-off. See ../THEORY.md for the derivation and GPU-mapping discussion.
//
// READ THIS AFTER: kernels.cuh (declarations + the thread-mapping idea).
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// Threads per block. 128 is a solid default on sm_75..sm_89 for a register-heavy
// per-thread ODE integrator: it is a multiple of the 32-lane warp, gives the
// scheduler 4 warps to hide latency, and keeps register pressure low enough for
// good occupancy. (Tune per GPU; see THEORY section "GPU mapping".)
static constexpr int THREADS_PER_BLOCK = 128;

// ---------------------------------------------------------------------------
// perfusion_kernel: thread idx owns sinusoid idx.
//   Launch config (set in integrate_gpu):
//     grid  = ceil(nsin / THREADS_PER_BLOCK) blocks
//     block = THREADS_PER_BLOCK threads
//   Thread-to-data map: idx = blockIdx.x * blockDim.x + threadIdx.x -> sinusoid idx.
//   Memory: LobuleConfig arrives by value (in registers/constant-ish arg space);
//   the whole RK4 march runs in registers/local memory; the only global write is
//   one SinusoidResult. No shared memory or atomics -- the members are fully
//   independent (embarrassing parallelism). Divergence is minimal: every thread
//   runs the same nseg steps; only the C<0 floor branch can differ.
// ---------------------------------------------------------------------------
__global__ void perfusion_kernel(LobuleConfig c, SinusoidResult* __restrict__ out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= lobule_size(c)) return;               // guard the ragged last block

    const double v = sinusoid_velocity(c, idx);      // this thread's inlet blood velocity
    out[idx] = integrate_sinusoid(c.p, v);           // shared physics -> matches the CPU
}

// ---------------------------------------------------------------------------
// integrate_gpu: host wrapper. Canonical CUDA steps:
//   (1) allocate the device output array
//   (2) launch one thread per sinusoid (config passed by value -- no input copy)
//   (3) copy results device->host
//   (4) free device memory
// We time ONLY the kernel with CUDA events so the figure is compute cost, not the
// tiny result copy (transfers are discussed separately in THEORY).
// ---------------------------------------------------------------------------
void integrate_gpu(const LobuleConfig& c, std::vector<SinusoidResult>& results, float* kernel_ms) {
    const int M = lobule_size(c);
    results.assign(static_cast<std::size_t>(M), SinusoidResult{});

    // (1) One SinusoidResult per sinusoid on the device. There are no per-member
    //     INPUT arrays to copy: each thread derives its velocity from LobuleConfig
    //     (passed by value), so the only device buffer is the output.
    SinusoidResult* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, static_cast<std::size_t>(M) * sizeof(SinusoidResult)));

    // (2) Launch. Ceiling division rounds the block count up to cover all M.
    const int blocks = (M + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    GpuTimer timer;
    timer.start();
    perfusion_kernel<<<blocks, THREADS_PER_BLOCK>>>(c, d_out);
    *kernel_ms = timer.stop_ms();                    // GPU-measured kernel time
    CUDA_CHECK_LAST("perfusion_kernel");             // catch launch + execution errors

    // (3) Bring the results back to the host vector.
    CUDA_CHECK(cudaMemcpy(results.data(), d_out,
                          static_cast<std::size_t>(M) * sizeof(SinusoidResult),
                          cudaMemcpyDeviceToHost));

    // (4) Always free what we allocated (no GPU garbage collector exists).
    CUDA_CHECK(cudaFree(d_out));
}
