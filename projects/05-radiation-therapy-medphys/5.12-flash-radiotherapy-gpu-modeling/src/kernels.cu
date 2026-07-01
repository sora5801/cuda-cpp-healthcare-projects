// ===========================================================================
// src/kernels.cu  --  Ensemble FLASH-chemistry kernel (one thread per voxel)
// ---------------------------------------------------------------------------
// Project 5.12 : FLASH Radiotherapy GPU Modeling
//
// WHAT THIS FILE DOES
//   Implements the device kernel (ensemble_kernel) and the host-side glue
//   (integrate_gpu) that allocates the result buffer, launches the kernel, times
//   it with CUDA events, and copies the results back. This is the GPU twin of
//   integrate_cpu() in reference_cpu.cpp: each GPU thread runs the SAME
//   integrate_voxel() from flash.h that the CPU loops over, so the two agree to
//   round-off. main.cu runs both and compares them per member.
//
//   There are NO device pointers for inputs: every member's parameters are
//   derived on the fly from the small EnsembleConfig (passed by value) via
//   member_job(), so the only device allocation is the output array. This keeps
//   the ensemble pattern's memory story simple -- pure compute, one write each.
//
// READ THIS AFTER: kernels.cuh (declarations + the thread-mapping idea), flash.h.
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// Threads per block. 128 is a solid default here: each thread runs a long,
// register-heavy RK4 time loop (double precision), so we favour a smaller block
// that keeps register pressure per SM manageable while still giving the
// scheduler several warps to hide latency. (256 also works; tune per GPU.)
static constexpr int THREADS_PER_BLOCK = 128;

// ---------------------------------------------------------------------------
// ensemble_kernel: thread idx owns ensemble member idx.
//   Launch config (set in integrate_gpu):
//     grid  = ceil(M / THREADS_PER_BLOCK) blocks, M = ensemble_size(c)
//     block = THREADS_PER_BLOCK threads
//   Thread-to-data map: idx = blockIdx.x * blockDim.x + threadIdx.x  ->  the
//   flat ensemble-member index (a (pO2, delivery-mode) pair, see member_axes).
//   Memory: reads only the by-value config (in local memory), writes one
//   VoxelResult to global memory. No shared memory, no atomics -- the members
//   are fully independent, which is exactly why this parallelises perfectly.
//   Divergence is mild: every member runs the same fixed number of RK4 steps;
//   only the min-O2 comparison branch differs, which is cheap.
// ---------------------------------------------------------------------------
__global__ void ensemble_kernel(EnsembleConfig c, VoxelResult* __restrict__ out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ensemble_size(c)) return;      // guard the ragged last block

    // Build this member's per-voxel job (pO2 + delivery mode) exactly as the CPU
    // does, then run the full pulse-train + relaxation integration. The heavy
    // lifting lives in flash.h so CPU and GPU share the identical arithmetic.
    const VoxelJob j = member_job(c, idx);
    out[idx] = integrate_voxel(j);
}

// ---------------------------------------------------------------------------
// integrate_gpu: host wrapper. Only ONE device buffer is needed (the outputs),
// because inputs are reconstructed inside the kernel from the by-value config.
// Steps: (1) size the host vector, (2) allocate d_out, (3) launch + time,
// (4) copy results D->H, (5) free. We time only step (3) with CUDA events so the
// figure is the kernel cost, not any transfer cost.
// ---------------------------------------------------------------------------
void integrate_gpu(const EnsembleConfig& c, std::vector<VoxelResult>& results,
                   float* kernel_ms) {
    const int M = ensemble_size(c);
    results.assign(static_cast<std::size_t>(M), VoxelResult{});   // (1)

    // (2) Only the outputs live in device memory (d_ prefix = DEVICE pointer;
    //     dereferencing it on the host would crash -- naming makes that visible).
    VoxelResult* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, static_cast<std::size_t>(M) * sizeof(VoxelResult)));

    // (3) Launch one thread per member; blocks cover M via ceiling division.
    const int blocks = (M + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    GpuTimer timer;
    timer.start();
    ensemble_kernel<<<blocks, THREADS_PER_BLOCK>>>(c, d_out);
    *kernel_ms = timer.stop_ms();             // GPU-measured kernel time
    CUDA_CHECK_LAST("ensemble_kernel");       // catch launch + execution errors

    // (4) Bring the results back to the host vector.
    CUDA_CHECK(cudaMemcpy(results.data(), d_out,
                          static_cast<std::size_t>(M) * sizeof(VoxelResult),
                          cudaMemcpyDeviceToHost));

    // (5) Free the one device allocation (no GPU garbage collector exists).
    CUDA_CHECK(cudaFree(d_out));
}
