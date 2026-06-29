// ===========================================================================
// src/kernels.cu  --  The GPU kernel and its host wrapper (MM-GBSA rescoring)
// ---------------------------------------------------------------------------
// Project 1.27 : MM-GBSA / MM-PBSA Rescoring
//
// WHAT THIS FILE DOES
//   Implements the device kernel (rescore_kernel) and the host-side glue
//   (rescore_gpu) that uploads the receptor + ligand snapshots, launches the
//   kernel, times it, and brings the per-snapshot energies back. This is the GPU
//   twin of rescore_cpu() in reference_cpu.cpp; main.cu runs both and compares
//   them. Because BOTH call the SAME snapshot_dg() from reference_cpu.h, the
//   per-snapshot arithmetic is identical (shared HD core, PATTERNS.md §2).
//
//   Pattern: independent jobs / one-thread-per-snapshot (PATTERNS.md §1). No
//   shared memory, no atomics -- each thread writes one dg[s]. The only memory
//   the threads share (read-only) is the receptor, which every thread scans in
//   full; we discuss the constant-memory alternative below.
//
// READ THIS AFTER: kernels.cuh (declarations + the thread-mapping idea).
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// Threads per block. 128 is a solid default here on sm_75..sm_89: each thread
// does a fairly heavy O(R*L) double-precision loop with many registers live, so
// a slightly smaller block (vs. 256) keeps register pressure from capping
// occupancy while still giving the scheduler 4 warps to hide latency. (The exact
// best value is GPU-specific; this is a teaching default, not a tuned one.)
static constexpr int THREADS_PER_BLOCK = 128;

// ---------------------------------------------------------------------------
// rescore_kernel: one thread evaluates one snapshot's binding-energy estimate.
//
//   Launch config (set in rescore_gpu):
//     grid  = min(ceil(S / THREADS_PER_BLOCK), CAP) blocks
//     block = THREADS_PER_BLOCK threads
//   Thread-to-data map: a grid-stride loop, so thread (blockIdx.x, threadIdx.x)
//   handles snapshots s = base, base+stride, base+2*stride, ... where
//   base = blockIdx.x*blockDim.x + threadIdx.x and stride = total thread count.
//   This lets a fixed-size grid cover an arbitrarily large S, and every snapshot
//   is handled by exactly one thread.
//
//   Memory:
//     * receptor and ligand are read from GLOBAL memory. The receptor is read by
//       EVERY thread and never written -- a classic __constant__ memory candidate
//       (its broadcast cache would serve a warp in one transaction). We keep it
//       in global memory because the receptor size is data-dependent (constant
//       memory is a fixed 64 KB bank), and the L1/L2 cache already serves the
//       repeated reads well at teaching sizes. THEORY §GPU mapping spells out the
//       trade-off and when constant memory wins.
//     * dg is written once per snapshot -- coalesced when consecutive threads in a
//       warp own consecutive snapshots (they do, by the base index formula).
//   No shared memory, no atomics: outputs are fully independent.
// ---------------------------------------------------------------------------
__global__ void rescore_kernel(const Atom* __restrict__ receptor, int R,
                               const Atom* __restrict__ ligand,   int L,
                               int S, double minus_TdS,
                               double* __restrict__ dg) {
    const int stride = blockDim.x * gridDim.x;          // total threads in grid
    for (int s = blockIdx.x * blockDim.x + threadIdx.x; // this thread's first snapshot
         s < S; s += stride) {
        // Base pointer to snapshot s's L ligand atoms in the flat device array.
        // This index arithmetic is IDENTICAL to rescore_cpu()'s, so the same
        // atoms are summed in the same order on both sides -> CPU==GPU.
        const Atom* lig_s = ligand + static_cast<std::size_t>(s) * L;
        // Call the SHARED physics. On the device this expands to the GPU build of
        // snapshot_dg(); on the host the same source built the reference. One line
        // here = the entire MM-GBSA energy for this frame.
        dg[s] = snapshot_dg(receptor, R, lig_s, L, minus_TdS);
    }
}

// ---------------------------------------------------------------------------
// rescore_gpu: host wrapper. The five canonical steps of a CUDA computation:
//   (1) allocate device memory  (2) copy inputs host->device
//   (3) launch the kernel        (4) copy result device->host
//   (5) free device memory
// We time ONLY step (3) with CUDA events so the reported figure is the kernel
// cost, not the PCIe transfer cost (transfers are discussed separately in
// THEORY §GPU mapping). The receptor and the entire flattened snapshot array are
// uploaded once; the per-snapshot outputs come back as a single contiguous copy.
// ---------------------------------------------------------------------------
void rescore_gpu(const Complex& cx, std::vector<double>& dg, float* kernel_ms) {
    const int    S          = cx.S;
    const std::size_t recBytes = static_cast<std::size_t>(cx.R) * sizeof(Atom);
    const std::size_t ligBytes = cx.ligand_snapshots.size() * sizeof(Atom);
    const std::size_t outBytes = static_cast<std::size_t>(S) * sizeof(double);

    dg.assign(static_cast<std::size_t>(S), 0.0);

    // (1) Device buffers. The d_ prefix marks DEVICE pointers (CLAUDE.md §12):
    //     dereferencing one on the host would crash, so the naming matters.
    Atom*   d_receptor = nullptr;   // [R] receptor atoms
    Atom*   d_ligand   = nullptr;   // [S*L] ligand atoms, row-major by snapshot
    double* d_dg       = nullptr;   // [S] per-snapshot energies (output)
    CUDA_CHECK(cudaMalloc(&d_receptor, recBytes));   // can fail: out of device memory
    CUDA_CHECK(cudaMalloc(&d_ligand,   ligBytes));
    CUDA_CHECK(cudaMalloc(&d_dg,       outBytes));

    // (2) Copy inputs H2D. .data() is the contiguous backing array of the vector;
    //     the flat ligand layout means one memcpy moves all snapshots at once.
    CUDA_CHECK(cudaMemcpy(d_receptor, cx.receptor.data(),         recBytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ligand,   cx.ligand_snapshots.data(), ligBytes,
                          cudaMemcpyHostToDevice));

    // (3) Launch. Enough blocks to give each snapshot a thread, capped so the grid
    //     stays modest; the grid-stride loop in the kernel covers any larger S.
    int blocks = (S + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;   // round up
    if (blocks < 1)    blocks = 1;
    if (blocks > 1024) blocks = 1024;   // cap: grid-stride handles the remainder
    GpuTimer timer;
    timer.start();
    rescore_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        d_receptor, cx.R, d_ligand, cx.L, S, cx.minus_TdS, d_dg);
    *kernel_ms = timer.stop_ms();        // GPU-measured kernel time
    CUDA_CHECK_LAST("rescore_kernel");   // catch launch + execution errors

    // (4) Bring the per-snapshot energies back to the host vector.
    CUDA_CHECK(cudaMemcpy(dg.data(), d_dg, outBytes, cudaMemcpyDeviceToHost));

    // (5) Always free what we allocated (no GPU garbage collector exists).
    CUDA_CHECK(cudaFree(d_receptor));
    CUDA_CHECK(cudaFree(d_ligand));
    CUDA_CHECK(cudaFree(d_dg));
}
