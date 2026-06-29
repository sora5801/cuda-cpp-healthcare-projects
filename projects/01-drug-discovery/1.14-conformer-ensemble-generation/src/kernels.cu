// ===========================================================================
// src/kernels.cu  --  GPU conformer-energy kernel + host wrapper
// ---------------------------------------------------------------------------
// Project 1.14 : Conformer Ensemble Generation
//
// This is the GPU twin of enumerate_energies_cpu() in reference_cpu.cpp. main.cu
// runs both and asserts they agree. The actual physics (index -> torsions -> 3D
// coordinates -> energy) lives in conformer.h as __host__ __device__ inline
// functions, so the kernel below is mostly the parallel BOOKKEEPING -- mapping
// threads to conformers -- around a single call to conformer_energy().
// See ../THEORY.md "GPU mapping".
// ===========================================================================
#include "conformer.h"           // conformer_energy (the shared HD physics)
#include "kernels.cuh"
#include "util/cuda_check.cuh"    // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"         // GpuTimer

// 128 threads/block. A multiple of the 32-lane warp, and a good fit here: each
// thread holds a small fixed work set in registers/local memory (the N_ATOMS=8
// position array + scratch), so a moderate block size keeps register pressure low
// while still giving the scheduler several warps to hide the latency of the cos/
// sin/sqrt the embedding does. (Tune per GPU; see THEORY "GPU mapping".)
static constexpr int THREADS_PER_BLOCK = 128;

// ---------------------------------------------------------------------------
// energies_kernel: one logical thread per conformer, via a grid-stride loop so a
// fixed-size grid still covers an arbitrarily large conformer count.
//   Launch config (set in energies_gpu):
//     grid  = min(ceil(n / THREADS_PER_BLOCK), cap) blocks
//     block = THREADS_PER_BLOCK threads
//   Thread-to-data map: thread (blockIdx.x, threadIdx.x) starts at
//     c = blockIdx.x * blockDim.x + threadIdx.x  and strides by the total thread
//     count until c >= n -- so every conformer index is handled exactly once.
//   Memory: each thread builds this conformer's coordinates entirely in its own
//     registers/local memory (conformer_energy allocates a tiny N_ATOMS array on
//     the stack); the ONLY global-memory traffic is the single out[c] store, which
//     is coalesced because consecutive threads write consecutive indices. No
//     shared memory and no atomics are needed -- the conformers are independent.
// ---------------------------------------------------------------------------
__global__ void energies_kernel(long n, double* __restrict__ out) {
    const long stride = static_cast<long>(blockDim.x) * gridDim.x;   // total threads
    for (long c = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
         c < n; c += stride) {
        // The entire per-conformer computation is one call into the shared core.
        // out_pos = nullptr: we want only the scalar energy here (the GPU does not
        // need to return coordinates; the CPU clustering step rebuilds them).
        out[c] = conformer_energy(c, nullptr);
    }
}

// ---------------------------------------------------------------------------
// energies_gpu: the five canonical CUDA steps, minus an input copy (there is no
// input array -- a conformer is fully described by its integer index, which the
// kernel derives from its thread id). We time ONLY the kernel (CUDA events), not
// the D2H copy (that is discussed separately in THEORY "GPU mapping").
//   (1) allocate the device output  (2) [no inputs to copy]
//   (3) launch the kernel            (4) copy energies D2H
//   (5) free device memory
// ---------------------------------------------------------------------------
void energies_gpu(std::vector<double>& energy, float* kernel_ms) {
    const long n = N_CONFORMER;
    energy.assign(static_cast<std::size_t>(n), 0.0);
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(double);

    // (1) Device output buffer. The d_ prefix marks a DEVICE pointer (CLAUDE.md
    //     §12): dereferencing it on the host would crash, so the naming matters.
    double* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, bytes));   // can fail: out of device memory

    // (3) Launch. Enough blocks to cover n one-thread-per-conformer, capped so the
    //     grid stays modest; the grid-stride loop in the kernel covers any larger n.
    int blocks = static_cast<int>((n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    if (blocks > 1024) blocks = 1024;        // cap: grid-stride handles the rest
    GpuTimer timer;
    timer.start();
    energies_kernel<<<blocks, THREADS_PER_BLOCK>>>(n, d_out);
    *kernel_ms = timer.stop_ms();            // GPU-measured kernel time
    CUDA_CHECK_LAST("energies_kernel");      // catch launch + execution errors

    // (4) Bring the energies back to the host vector, then (5) free the buffer.
    CUDA_CHECK(cudaMemcpy(energy.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_out));
}
