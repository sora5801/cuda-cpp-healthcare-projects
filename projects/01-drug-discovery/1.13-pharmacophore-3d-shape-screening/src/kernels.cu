// ===========================================================================
// src/kernels.cu  --  Gaussian shape-overlap screening kernel + host wrapper
// ---------------------------------------------------------------------------
// Project 1.13 : Pharmacophore & 3D Shape Screening
//
// This is the GPU twin of shape_tanimoto_cpu() in reference_cpu.cpp. main.cu
// runs both and asserts they agree. Both sides call the SAME __host__ __device__
// physics in shape_overlap.h, so agreement is to ~machine precision, not just
// "close" (PATTERNS.md sec 2 + sec 4). See ../THEORY.md sec 4 for the mapping.
// ===========================================================================
#include "kernels.cuh"
#include "shape_overlap.h"       // molecule_overlap, shape_tanimoto, Molecule
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer

// ---------------------------------------------------------------------------
// The QUERY molecule in CONSTANT memory.
//   * Every thread reads ALL of the query's atoms (the inner loop of
//     molecule_overlap walks them) but NONE writes them, and they are identical
//     for the whole launch -> constant memory is the ideal home: its hardware
//     cache broadcasts one address to an entire warp in a single transaction,
//     instead of each thread issuing its own global loads for the query atoms.
//   * Size is fixed at compile time: sizeof(Molecule) = 4 + 64*32 = 2052 bytes
//     (one int + MAX_ATOMS atoms of 4 doubles), well within the 64 KB constant
//     bank. Filled by cudaMemcpyToSymbol() in shape_screen_gpu().
//   * It is a full Molecule (not a pointer), so we can hand it straight to the
//     shared molecule_overlap(const Molecule&, ...) by reference.
// ---------------------------------------------------------------------------
__constant__ Molecule c_query;

// 128 threads/block: a multiple of the 32-lane warp. This kernel is COMPUTE-
// heavy (a double loop of exp() per thread) and register-hungry (each thread
// holds a Molecule's worth of fit-atom coordinates in flight), so a smaller
// block than the usual 256 keeps register pressure down without hurting the
// scheduler's ability to hide latency. See THEORY sec 4 "occupancy".
static constexpr int THREADS_PER_BLOCK = 128;

// ---------------------------------------------------------------------------
// shape_screen_kernel: one logical thread per library conformer, via a grid-
// stride loop so a fixed-size grid still covers an arbitrarily large library.
//   Thread (blockIdx.x, threadIdx.x) starts at k = block*blockDim + thread and
//   strides by the total thread count until k >= n.
//   Memory: c_query from the constant cache; conformer k (d_lib[k]) from global
//   memory. No shared memory or atomics: the per-conformer scores are fully
//   independent, so there is nothing to synchronize -- the cleanest possible
//   parallel pattern.
//   Work per thread: O(M*K) Gaussian evaluations (M query atoms, K fit atoms),
//   computed by the SAME molecule_overlap() the CPU uses -> identical results.
// ---------------------------------------------------------------------------
__global__ void shape_screen_kernel(const Molecule* __restrict__ d_lib, int n,
                                    double o_aa, double* __restrict__ d_out) {
    const int stride = blockDim.x * gridDim.x;              // total threads in grid
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < n; k += stride) {
        // Load this thread's conformer once into a register/local copy. Passing
        // it by reference into molecule_overlap lets the compiler keep its atoms
        // in registers across the inner loop.
        const Molecule& B = d_lib[k];

        // Cross overlap O_AB (query vs this conformer) and the fit self-overlap
        // O_BB. The query self-overlap O_AA was computed once on the host and
        // passed in -- recomputing it per thread would be pure wasted work.
        const double o_ab = molecule_overlap(c_query, B);
        const double o_bb = molecule_overlap(B, B);

        // Same exact ratio as the CPU reference -> bit-for-bit comparable.
        d_out[k] = shape_tanimoto(o_ab, o_aa, o_bb);
    }
}

// ---------------------------------------------------------------------------
// shape_screen_gpu: the canonical CUDA steps, with the query going to constant
// memory and O_AA hoisted to the host. We time ONLY the kernel (CUDA events),
// not the H2D/D2H copies (those are discussed separately in THEORY sec 4).
// ---------------------------------------------------------------------------
void shape_screen_gpu(const ConformerSet& set, std::vector<double>& out, float* kernel_ms) {
    const int n = set.n;
    out.assign(static_cast<std::size_t>(n), 0.0);
    const std::size_t lib_bytes = static_cast<std::size_t>(n) * sizeof(Molecule);
    const std::size_t out_bytes = static_cast<std::size_t>(n) * sizeof(double);

    // (a) Upload the query to the __constant__ symbol (a special copy that
    //     targets the constant bank rather than ordinary global memory).
    CUDA_CHECK(cudaMemcpyToSymbol(c_query, &set.query, sizeof(Molecule)));

    // (b) Precompute the query self-overlap O_AA ONCE on the host. It is a
    //     single small molecule_overlap() and is identical for every conformer,
    //     so doing it here (instead of inside every thread) is a clear win and
    //     keeps the kernel's per-thread work to the two conformer-dependent
    //     overlaps. This mirrors shape_tanimoto_cpu() exactly.
    const double o_aa = molecule_overlap(set.query, set.query);

    // (c) Allocate + upload the library (one contiguous block of Molecules) and
    //     allocate the output scores.
    Molecule* d_lib = nullptr;   // [n] device, POD Molecules
    double*   d_out = nullptr;   // [n] device scores
    CUDA_CHECK(cudaMalloc(&d_lib, lib_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, out_bytes));
    CUDA_CHECK(cudaMemcpy(d_lib, set.lib.data(), lib_bytes, cudaMemcpyHostToDevice));

    // (d) Launch. Enough blocks to cover n one-thread-per-conformer, but capped
    //     so the grid stays modest; the grid-stride loop handles any remainder.
    int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (blocks > 1024) blocks = 1024;   // cap: grid-stride covers any larger n
    GpuTimer timer;
    timer.start();
    shape_screen_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_lib, n, o_aa, d_out);
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("shape_screen_kernel");

    // (e) Copy scores back, then free device memory.
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_lib));
    CUDA_CHECK(cudaFree(d_out));
}
