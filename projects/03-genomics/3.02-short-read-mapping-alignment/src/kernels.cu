// ===========================================================================
// src/kernels.cu  --  Seed-and-extend mapping kernel + its host wrapper
// ---------------------------------------------------------------------------
// Project 3.2 : Short-Read Mapping / Alignment
//
// WHAT THIS FILE DOES
//   Implements the device kernel (map_reads_kernel) and the host-side glue
//   (map_reads_gpu) that allocates GPU memory, uploads the reference + reads +
//   k-mer index, launches the kernel, times it with CUDA events, and copies the
//   per-read results back. This is the GPU twin of map_reads_cpu() in
//   reference_cpu.cpp; main.cu runs both and asserts every read matches exactly.
//
//   THE KEY POINT: the kernel does NOT re-derive any scoring math. It calls the
//   very same `__host__ __device__` helpers the CPU reference uses --
//   kmer_code(), kmer_equal_range(), score_window() from reference_cpu.h -- so a
//   thread's arithmetic is byte-identical to the CPU loop. That is what makes the
//   GPU result verifiable by exact integer equality (PATTERNS.md sections 2, 4).
//
// READ THIS AFTER: kernels.cuh (declarations + the one-thread-per-read idea).
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// Threads per block. 256 is a solid default on sm_75..sm_89: a multiple of the
// 32-lane warp, it gives the scheduler 8 warps per block to hide the global-
// memory latency of the index binary search and the read scan, while leaving
// plenty of blocks resident for occupancy. (Tune per GPU; see THEORY section 4.)
static constexpr int THREADS_PER_BLOCK = 256;

// ---------------------------------------------------------------------------
// map_reads_kernel: ONE THREAD MAPS ONE READ (grid-stride loop over all reads).
//
//   Launch config (set in map_reads_gpu):
//     grid  = min(1024, ceil(n_reads / THREADS_PER_BLOCK)) blocks
//     block = THREADS_PER_BLOCK threads
//   Thread-to-data map: a thread starts at r = blockIdx.x*blockDim.x+threadIdx.x
//   and strides by the total thread count, so a fixed-size grid still covers an
//   arbitrarily large read batch (the grid-stride-loop idiom, cf. 1.12).
//
//   Memory: each thread reads its read's row + the reference + the shared sorted
//   index from GLOBAL memory, and keeps its running-best (pos, score, mism) in
//   REGISTERS. No shared memory and NO ATOMICS: the per-read max-reduction is
//   private to the thread, and the outputs are disjoint (thread r writes only
//   slot r). This is the simplest correct structure -- and being atomic-free, it
//   is automatically deterministic (PATTERNS.md section 3).
//
//   Tie-break MUST match the CPU exactly: highest score wins; on a tie the
//   LOWEST reference offset. We scan candidate offsets in the index's ascending-
//   offset order and replace the best only on a STRICT improvement (sc > best),
//   so the first (lowest) offset achieving the max is kept -- identical to
//   map_one() in reference_cpu.cpp.
// ---------------------------------------------------------------------------
__global__ void map_reads_kernel(const uint8_t* __restrict__ ref, int ref_len,
                                 const uint8_t* __restrict__ reads, int read_len,
                                 int n_reads,
                                 const uint64_t* __restrict__ sorted_codes,
                                 const int* __restrict__ sorted_offsets,
                                 int n_kmers,
                                 int* __restrict__ out_pos,
                                 int* __restrict__ out_score,
                                 int* __restrict__ out_mism) {
    const int stride = blockDim.x * gridDim.x;                  // total threads
    for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < n_reads; r += stride) {
        // Pointer to this thread's read row in the flat reads buffer.
        const uint8_t* read = reads + static_cast<std::size_t>(r) * read_len;

        // ---- SEED: this read's leading k-mer, looked up in the sorted index --
        const uint64_t qcode = kmer_code(read, 0);   // bases [0, SEED_K)
        int lo = 0, hi = 0;
        kmer_equal_range(sorted_codes, n_kmers, qcode, &lo, &hi);

        // ---- EXTEND: score the read at every candidate offset; keep the best --
        int best_pos   = NO_HIT;
        int best_score = -2000000;   // below any real/off-end score so the first
                                     // real candidate always wins
        int best_mism  = read_len;
        for (int i = lo; i < hi; ++i) {
            const int pos = sorted_offsets[i];       // candidate reference offset
            int mism = 0;
            const int sc = score_window(ref, ref_len, read, read_len, pos, &mism);
            if (sc > best_score) {                   // strict > => lowest-offset tie winner
                best_score = sc;
                best_pos   = pos;
                best_mism  = mism;
            }
        }

        // Normalize a complete miss (seed found nothing) to a clean 0 score,
        // exactly as the CPU reference does, so the two agree on misses too.
        if (best_pos == NO_HIT) { best_score = 0; best_mism = 0; }

        // Write this read's result (disjoint slot -> no race).
        out_pos[r]   = best_pos;
        out_score[r] = best_score;
        out_mism[r]  = best_mism;
    }
}

// ---------------------------------------------------------------------------
// map_reads_gpu: host wrapper. The canonical CUDA steps, adapted to several
// input buffers:
//   (1) allocate device memory for ref, reads, index, and the 3 output arrays
//   (2) copy all inputs host->device
//   (3) launch map_reads_kernel
//   (4) copy the 3 result arrays device->host and pack into MapResult[]
//   (5) free device memory
// We time ONLY step (3) with CUDA events so the reported figure is the kernel
// cost, not the PCIe transfer cost (transfers are discussed in THEORY section 4).
// ---------------------------------------------------------------------------
void map_reads_gpu(const MappingProblem& prob, const KmerIndex& index,
                   std::vector<MapResult>& results, float* kernel_ms) {
    const int R = prob.n_reads;
    results.assign(static_cast<std::size_t>(R), MapResult{});

    // Byte sizes of each buffer (note ref/reads are uint8_t = 1 byte/base).
    const std::size_t ref_bytes   = static_cast<std::size_t>(prob.ref_len);
    const std::size_t reads_bytes = static_cast<std::size_t>(R) * prob.read_len;
    const std::size_t code_bytes  = static_cast<std::size_t>(index.n_kmers) * sizeof(uint64_t);
    const std::size_t off_bytes   = static_cast<std::size_t>(index.n_kmers) * sizeof(int);
    const std::size_t out_bytes   = static_cast<std::size_t>(R) * sizeof(int);

    // (1) Device buffers. d_ prefix = DEVICE pointer (dereferencing on the host
    //     would crash). Each cudaMalloc can fail (out of device memory).
    uint8_t  *d_ref = nullptr, *d_reads = nullptr;
    uint64_t *d_codes = nullptr;
    int      *d_off = nullptr, *d_pos = nullptr, *d_score = nullptr, *d_mism = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ref,   ref_bytes));
    CUDA_CHECK(cudaMalloc(&d_reads, reads_bytes));
    CUDA_CHECK(cudaMalloc(&d_codes, code_bytes));
    CUDA_CHECK(cudaMalloc(&d_off,   off_bytes));
    CUDA_CHECK(cudaMalloc(&d_pos,   out_bytes));
    CUDA_CHECK(cudaMalloc(&d_score, out_bytes));
    CUDA_CHECK(cudaMalloc(&d_mism,  out_bytes));

    // (2) Copy all inputs H2D. .data() is the contiguous backing array.
    CUDA_CHECK(cudaMemcpy(d_ref,   prob.ref.data(),            ref_bytes,   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_reads, prob.reads.data(),          reads_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_codes, index.sorted_codes.data(),  code_bytes,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_off,   index.sorted_offsets.data(),off_bytes,   cudaMemcpyHostToDevice));

    // (3) Launch. Enough blocks to cover all reads one-thread-each, capped at
    //     1024 blocks; the grid-stride loop in the kernel handles any larger R.
    int blocks = (R + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (blocks < 1)    blocks = 1;       // never launch zero blocks
    if (blocks > 1024) blocks = 1024;    // cap; grid-stride covers the remainder
    GpuTimer timer;
    timer.start();
    map_reads_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        d_ref, prob.ref_len, d_reads, prob.read_len, R,
        d_codes, d_off, index.n_kmers, d_pos, d_score, d_mism);
    *kernel_ms = timer.stop_ms();        // GPU-measured kernel time
    CUDA_CHECK_LAST("map_reads_kernel"); // catch launch + execution errors

    // (4) Copy the three result arrays back and pack them into MapResult[].
    std::vector<int> h_pos(static_cast<std::size_t>(R));
    std::vector<int> h_score(static_cast<std::size_t>(R));
    std::vector<int> h_mism(static_cast<std::size_t>(R));
    CUDA_CHECK(cudaMemcpy(h_pos.data(),   d_pos,   out_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_score.data(), d_score, out_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_mism.data(),  d_mism,  out_bytes, cudaMemcpyDeviceToHost));
    for (int r = 0; r < R; ++r) {
        results[static_cast<std::size_t>(r)].pos   = h_pos[static_cast<std::size_t>(r)];
        results[static_cast<std::size_t>(r)].score = h_score[static_cast<std::size_t>(r)];
        results[static_cast<std::size_t>(r)].mism  = h_mism[static_cast<std::size_t>(r)];
    }

    // (5) Always free what we allocated (there is no GPU garbage collector).
    CUDA_CHECK(cudaFree(d_ref));
    CUDA_CHECK(cudaFree(d_reads));
    CUDA_CHECK(cudaFree(d_codes));
    CUDA_CHECK(cudaFree(d_off));
    CUDA_CHECK(cudaFree(d_pos));
    CUDA_CHECK(cudaFree(d_score));
    CUDA_CHECK(cudaFree(d_mism));
}
