// ===========================================================================
// src/kernels.cu  --  The GPU classifier kernel and its host wrapper
// ---------------------------------------------------------------------------
// Project 3.14 : Metagenomic Taxonomic Classification
//
// WHAT THIS FILE DOES
//   Implements the device kernel (classify_kernel) and the host-side glue
//   (classify_gpu) that allocates GPU memory, uploads the reads + reference
//   table, launches the kernel, times it, and brings the per-read taxon ids
//   back. This is the GPU twin of classify_cpu() in reference_cpu.cpp; main.cu
//   runs both and asserts they agree EXACTLY (integer taxon ids -> tolerance 0).
//
//   The actual per-read logic is NOT here -- it is the shared classify_read()
//   from kmer_core.h, which nvcc compiles for the device because it is marked
//   __host__ __device__. That single shared function is what guarantees CPU/GPU
//   parity (docs/PATTERNS.md sec 2). This file is just the parallel HARNESS.
//
// READ THIS AFTER: kmer_core.h (the math), kernels.cuh (the thread-mapping idea).
// ===========================================================================
#include "kernels.cuh"
#include "kmer_core.h"           // classify_read (__host__ __device__), MAX_TAXA
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// Threads per block. 256 is a solid default on sm_75..sm_89: a multiple of the
// 32-lane warp, enough warps to hide the latency of the data-dependent global
// look-ups, and plenty of resident blocks for occupancy. Each thread keeps a
// small per-read vote array (MAX_TAXA ints) in registers/local memory, so the
// register pressure stays modest at this block size.
static constexpr int THREADS_PER_BLOCK = 256;

// ---------------------------------------------------------------------------
// classify_kernel: one thread classifies one read, via a grid-stride loop so a
// fixed-size grid covers an arbitrarily large read set.
//   Thread (blockIdx.x, threadIdx.x) starts at read index
//     i = blockIdx.x * blockDim.x + threadIdx.x
//   and strides by the total thread count until i >= n_reads.
//   Memory:
//     * bases/offset/length : this read's slice, read from GLOBAL memory.
//     * keys/taxa           : the shared reference table in GLOBAL memory; the
//       probe addresses are data-dependent (hash of each k-mer), so the L2 cache
//       -- not constant memory -- is what accelerates repeated hits.
//     * votes[]             : per-thread scratch in registers/local memory.
//   No shared memory and NO ATOMICS: each thread writes only its own out[i], so
//   outputs are independent and the result is order-independent (deterministic).
// ---------------------------------------------------------------------------
__global__ void classify_kernel(const char* __restrict__ bases,
                                const int* __restrict__ offset,
                                const int* __restrict__ length,
                                int n_reads,
                                const uint64_t* __restrict__ keys,
                                const uint32_t* __restrict__ taxa,
                                uint64_t capacity,
                                uint32_t* __restrict__ out) {
    // Per-thread vote histogram. It lives in this thread's local stack (registers
    // if it fits) and is private to the read this thread is classifying -- so two
    // threads never contend, hence no atomics. classify_read() zeroes it.
    int votes[MAX_TAXA];

    const int stride = blockDim.x * gridDim.x;             // total threads in grid
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_reads; i += stride) {
        const char* read = bases + offset[i];              // this thread's read
        // Call the SHARED core -- byte-for-byte the same function classify_cpu()
        // runs on the host. That is why GPU == CPU exactly (see THEORY "verify").
        out[i] = classify_read(read, length[i], keys, taxa, capacity, votes);
    }
}

// ---------------------------------------------------------------------------
// classify_gpu: host wrapper. The five canonical steps of a CUDA computation:
//   (1) allocate device memory   (2) copy inputs host->device
//   (3) launch the kernel         (4) copy result device->host
//   (5) free device memory
// We time ONLY step (3) with CUDA events so the reported figure is the kernel
// cost, not the PCIe transfer cost (transfers are discussed in THEORY).
// ---------------------------------------------------------------------------
void classify_gpu(const ReadSet& reads, const RefDatabase& db,
                  std::vector<uint32_t>& out, float* kernel_ms) {
    const int      n_reads     = reads.n_reads;
    const std::size_t base_bytes = reads.bases.size() * sizeof(char);
    const std::size_t idx_bytes  = static_cast<std::size_t>(n_reads) * sizeof(int);
    const std::size_t out_bytes  = static_cast<std::size_t>(n_reads) * sizeof(uint32_t);
    const std::size_t key_bytes  = db.capacity * sizeof(uint64_t);
    const std::size_t tax_bytes  = db.capacity * sizeof(uint32_t);
    out.assign(static_cast<std::size_t>(n_reads), TAXON_UNCLASSIFIED);

    // (1) Device buffers. The d_ prefix marks DEVICE pointers (CLAUDE.md 12).
    char*     d_bases  = nullptr;   // [total_bases] concatenated reads
    int*      d_offset = nullptr;   // [n_reads] per-read start offsets
    int*      d_length = nullptr;   // [n_reads] per-read lengths
    uint64_t* d_keys   = nullptr;   // [capacity] reference table k-mers
    uint32_t* d_taxa   = nullptr;   // [capacity] reference table taxon ids
    uint32_t* d_out    = nullptr;   // [n_reads] output taxon ids
    CUDA_CHECK(cudaMalloc(&d_bases,  base_bytes));    // can fail: out of device memory
    CUDA_CHECK(cudaMalloc(&d_offset, idx_bytes));
    CUDA_CHECK(cudaMalloc(&d_length, idx_bytes));
    CUDA_CHECK(cudaMalloc(&d_keys,   key_bytes));
    CUDA_CHECK(cudaMalloc(&d_taxa,   tax_bytes));
    CUDA_CHECK(cudaMalloc(&d_out,    out_bytes));

    // (2) Copy inputs H2D. The reads and the whole reference table move once.
    CUDA_CHECK(cudaMemcpy(d_bases,  reads.bases.data(),  base_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offset, reads.offset.data(), idx_bytes,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_length, reads.length.data(), idx_bytes,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_keys,   db.keys.data(),      key_bytes,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_taxa,   db.taxa.data(),      tax_bytes,  cudaMemcpyHostToDevice));

    // (3) Launch. Enough blocks to cover n_reads one-thread-per-read, capped so
    //     the grid stays modest; the grid-stride loop in the kernel handles any
    //     remainder (so the same code scales from 16 reads to millions).
    int blocks = (n_reads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (blocks < 1)    blocks = 1;       // at least one block even for tiny inputs
    if (blocks > 1024) blocks = 1024;    // cap: grid-stride covers any larger n
    GpuTimer timer;
    timer.start();
    classify_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_bases, d_offset, d_length,
                                                   n_reads, d_keys, d_taxa,
                                                   db.capacity, d_out);
    *kernel_ms = timer.stop_ms();        // GPU-measured kernel time
    CUDA_CHECK_LAST("classify_kernel");  // catch launch + execution errors

    // (4) Bring the per-read taxon ids back to the host vector.
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));

    // (5) Always free what we allocated (no GPU garbage collector exists).
    CUDA_CHECK(cudaFree(d_bases));
    CUDA_CHECK(cudaFree(d_offset));
    CUDA_CHECK(cudaFree(d_length));
    CUDA_CHECK(cudaFree(d_keys));
    CUDA_CHECK(cudaFree(d_taxa));
    CUDA_CHECK(cudaFree(d_out));
}
