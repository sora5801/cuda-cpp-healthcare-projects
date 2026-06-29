// ===========================================================================
// src/kernels.cu  --  Virtual-screening kernel + host wrapper
// ---------------------------------------------------------------------------
// Project 1.4 : Ultra-Large Virtual Screening
//
// WHAT THIS FILE DOES
//   Implements the device kernel (screen_kernel) and the host glue (screen_gpu)
//   that uploads the library, launches the kernel, times it, and brings the
//   scores back. This is the GPU twin of screen_cpu() in reference_cpu.cpp; both
//   call the SAME shared score_ligand() from screen_core.h, so main.cu can run
//   both and assert they agree bit-for-bit. See ../THEORY.md "GPU mapping".
//
// READ THIS AFTER: kernels.cuh (declarations + the thread-mapping idea) and
// screen_core.h (the per-ligand math the kernel calls).
// ===========================================================================
#include "kernels.cuh"
#include "screen_core.h"         // Target, Ligand, score_ligand (shared HD core)
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// ---------------------------------------------------------------------------
// The screening TARGET in CONSTANT memory.
//   Every thread reads the same target (the binding-site wish list) and NONE
//   writes it during the launch -> constant memory is the ideal home: its
//   hardware cache broadcasts one address to a whole warp in a single
//   transaction, instead of every thread issuing its own global load. The struct
//   is tiny (4 ints = 16 bytes), trivially within the 64 KB constant bank.
//   Filled by cudaMemcpyToSymbol() in screen_gpu(). This mirrors how project
//   1.12 parks its query fingerprint in constant memory.
// ---------------------------------------------------------------------------
__constant__ Target c_target;

// Threads per block. 256 is a solid default on sm_75..sm_89: a multiple of the
// 32-lane warp, gives the scheduler 8 warps to hide memory latency, and leaves
// plenty of blocks resident for occupancy. (See THEORY "GPU mapping".)
static constexpr int THREADS_PER_BLOCK = 256;

// ---------------------------------------------------------------------------
// screen_kernel: one logical thread per ligand, via a grid-stride loop so a
// fixed-size grid covers an arbitrarily large library.
//   Launch config (set in screen_gpu): block = 256 threads; blocks chosen to
//   cover n but capped, with the grid-stride loop handling any remainder.
//   Thread-to-data map: thread starts at i = blockIdx.x*blockDim.x + threadIdx.x
//   and strides by the total thread count (blockDim.x*gridDim.x) until i >= n.
//   Memory: c_target from the constant cache; ligands[i] from global memory
//   (coalesced -- consecutive threads read consecutive Ligand structs). No shared
//   memory and no atomics: each output score[i] is fully independent.
// ---------------------------------------------------------------------------
__global__ void screen_kernel(const Ligand* __restrict__ ligands, int n,
                              int* __restrict__ score) {
    const int stride = blockDim.x * gridDim.x;             // total threads in grid
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        // The ENTIRE per-ligand pipeline (filter cascade + surrogate dock) is the
        // shared score_ligand() -- the very same function the CPU reference runs.
        // Because it is integer-only, this thread's result is bit-identical to the
        // CPU's for ligand i. Reading c_target by value copies the 16-byte struct
        // into registers once per iteration (cheap; it comes from the broadcast
        // constant cache, not global memory).
        score[i] = score_ligand(ligands[i], c_target);
    }
}

// ---------------------------------------------------------------------------
// screen_gpu: host wrapper. The canonical CUDA steps, with the target going to
// constant memory rather than a global buffer. We time ONLY the kernel (CUDA
// events), not the H2D/D2H copies (discussed separately in THEORY).
// ---------------------------------------------------------------------------
void screen_gpu(const LigandLibrary& lib, std::vector<int>& score, float* kernel_ms) {
    const int n = lib.n();
    score.assign(static_cast<std::size_t>(n), 0);
    const std::size_t lig_bytes   = static_cast<std::size_t>(n) * sizeof(Ligand);
    const std::size_t score_bytes = static_cast<std::size_t>(n) * sizeof(int);

    // (a) Upload the target to the __constant__ symbol (a special copy that
    //     targets the constant bank, not ordinary global memory).
    CUDA_CHECK(cudaMemcpyToSymbol(c_target, &lib.target, sizeof(Target)));

    // (b) Allocate + upload the ligand library, and allocate the output scores.
    //     The d_ prefix marks DEVICE pointers (CLAUDE.md sec 12): dereferencing
    //     one on the host would crash, so the naming convention matters.
    Ligand* d_ligands = nullptr;   // [n] device array of Ligand structs
    int*    d_score   = nullptr;   // [n] device array of scores (output)
    CUDA_CHECK(cudaMalloc(&d_ligands, lig_bytes));    // can fail: out of memory
    CUDA_CHECK(cudaMalloc(&d_score,   score_bytes));
    CUDA_CHECK(cudaMemcpy(d_ligands, lib.ligands.data(), lig_bytes,
                          cudaMemcpyHostToDevice));

    // (c) Launch. Enough blocks to cover n one-thread-per-ligand, but capped so
    //     the grid stays modest; the grid-stride loop covers any larger library.
    int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (blocks > 1024) blocks = 1024;   // cap: grid-stride handles n beyond this
    GpuTimer timer;
    timer.start();
    screen_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_ligands, n, d_score);
    *kernel_ms = timer.stop_ms();        // GPU-measured kernel time
    CUDA_CHECK_LAST("screen_kernel");    // catch launch + execution errors

    // (d) Copy scores back, then (e) free device memory (no GPU GC exists).
    CUDA_CHECK(cudaMemcpy(score.data(), d_score, score_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_ligands));
    CUDA_CHECK(cudaFree(d_score));
}
