// ===========================================================================
// src/kernels.cu  --  GPU knockout screen (one LP per thread) + host wrapper
// ---------------------------------------------------------------------------
// Project 6.12 : Metabolic Flux / Constraint-Based Modeling
//
// WHAT THIS FILE DOES
//   Implements the device kernel (screen_kernel) that solves one FBA linear
//   program per thread, and the host glue (screen_gpu) that allocates the result
//   buffer, launches the kernel, times it, and copies the answers back. It is the
//   GPU twin of screen_cpu() in reference_cpu.cpp; main.cu runs both and asserts
//   they agree bit-for-bit.
//
//   The heavy lifting -- the bounded-variable simplex -- is NOT here: it lives in
//   fba.h as __host__ __device__ code so both paths call the exact same solver.
//   This file only supplies the CUDA thread-mapping and memory plumbing.
//
// READ THIS AFTER: kernels.cuh (the ensemble-of-LPs idea) and fba.h (the solver).
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// Threads per block. 64 is a deliberate, modest choice here: each thread holds a
// large private simplex tableau in local memory (see fba.h), so packing 256
// threads/block would spill hard and hurt more than it helps. 64 keeps enough
// warps resident to hide latency while leaving register/local headroom. This is a
// case where the RIGHT block size is dictated by per-thread state, not by a
// one-size-fits-all default -- a genuine occupancy lesson (THEORY.md GPU mapping).
static constexpr int THREADS_PER_BLOCK = 64;

// ---------------------------------------------------------------------------
// screen_kernel: one thread solves one FBA LP.
//   Launch config (set in screen_gpu):
//     grid  = ceil(njobs / THREADS_PER_BLOCK) blocks
//     block = THREADS_PER_BLOCK threads
//   Thread-to-job map: k = blockIdx.x * blockDim.x + threadIdx.x, where
//     k in [0, nrxn)  -> solve with reaction k deleted (a knockout), and
//     k == nrxn       -> solve the wild type (njobs = nrxn + 1 total jobs).
//   Memory: `model` is a by-value copy in this thread's local memory; the solver
//   allocates its tableau in local memory too. No shared memory, no atomics, no
//   cross-thread communication -- pure independent parallelism over knockouts.
// ---------------------------------------------------------------------------
__global__ void screen_kernel(FbaModel model, FbaResult* __restrict__ out, int njobs) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's job
    if (k >= njobs) return;                                // guard the ragged last block

    // k in [0,nrxn) deletes reaction k; k == nrxn (== model.nrxn) => wild type.
    // solve_knockout treats any out-of-range ko as "no deletion", so passing
    // model.nrxn cleanly yields the wild-type solve.
    const int ko = (k < model.nrxn) ? k : -1;
    out[k] = solve_knockout(model, ko);   // the shared simplex from fba.h
}

// ---------------------------------------------------------------------------
// screen_gpu: host wrapper. The canonical CUDA steps, minus input H2D (there is
//   no big input array -- the model rides along in the kernel's by-value arg):
//   (1) allocate the device result buffer  (2) launch  (3) copy results D2H
//   (4) free. We time ONLY the launch with CUDA events so the figure is kernel
//   cost, not allocation/copy cost (discussed in THEORY.md).
// ---------------------------------------------------------------------------
void screen_gpu(const FbaModel& model, std::vector<FbaResult>& results, float* kernel_ms) {
    const int njobs = model.nrxn + 1;                     // knockouts + wild type
    results.assign(static_cast<std::size_t>(njobs), FbaResult{});
    const std::size_t bytes = static_cast<std::size_t>(njobs) * sizeof(FbaResult);

    // (1) Device buffer for the results (one FbaResult per job). d_ = device ptr.
    FbaResult* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, bytes));                // can fail: out of memory

    // (2) Launch: enough blocks to cover all njobs jobs (ceiling division).
    const int blocks = (njobs + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    GpuTimer timer;
    timer.start();
    screen_kernel<<<blocks, THREADS_PER_BLOCK>>>(model, d_out, njobs);
    *kernel_ms = timer.stop_ms();                         // GPU-measured kernel time
    CUDA_CHECK_LAST("screen_kernel");                     // catch launch + run errors

    // (3) Copy the (nrxn+1) results back to the host vector.
    CUDA_CHECK(cudaMemcpy(results.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    // (4) Release the device buffer (no GPU garbage collector).
    CUDA_CHECK(cudaFree(d_out));
}
