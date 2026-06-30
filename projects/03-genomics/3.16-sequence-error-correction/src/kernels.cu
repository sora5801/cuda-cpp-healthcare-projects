// ===========================================================================
// src/kernels.cu  --  The two GPU kernels + the host wrapper
// ---------------------------------------------------------------------------
// Project 3.16 : Sequence Error Correction  (k-mer spectrum / trusted-k-mer)
//
// WHAT THIS FILE DOES
//   Implements the two device kernels and the host glue:
//     count_kmers_kernel    -- phase 1: histogram the k-mer spectrum via atomics.
//     correct_reads_kernel  -- phase 2: correct each read via the shared physics.
//     correct_reads_gpu     -- allocate, upload, launch both, time, download.
//   Both kernels are the GPU twins of the serial loops in reference_cpu.cpp;
//   main.cu runs the CPU and GPU sides and asserts they are IDENTICAL (==),
//   which is possible because the per-element logic is the same __host__ __device__
//   code from reference_cpu.h.
//
// READ THIS AFTER: kernels.cuh (declarations + the two-phase idea), reference_cpu.h
// (the shared physics these kernels call).
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// Threads per block. 256 is a solid default on sm_75..sm_89: a multiple of the
// 32-lane warp, 8 warps to hide memory latency, plenty of resident blocks for
// occupancy. Both kernels are launched with one thread per READ (not per base),
// so the grid is ceil(n_reads / 256) -- small, since n_reads is modest here.
static constexpr int THREADS_PER_BLOCK = 256;

// ---------------------------------------------------------------------------
// count_kmers_kernel  (PHASE 1: build the spectrum)
//
//   Thread-to-data map: thread g = blockIdx.x*blockDim.x + threadIdx.x owns
//   READ g (guard g < n for the ragged last block). The thread slides a length-k
//   window across its read and, for each valid k-mer, does
//       atomicAdd(&d_counts[code], 1u);
//
//   WHY ATOMICS: many reads contain the SAME k-mer (that is the whole point --
//   true k-mers recur), so different threads race to bump the same slot. atomicAdd
//   serializes those bumps in hardware so none is lost. And because integer
//   addition COMMUTES (a+b == b+a), the final table does not depend on the order
//   the atomics happen to land -> the GPU spectrum is bit-identical to the serial
//   CPU spectrum every run (PATTERNS.md sec 3: integer atomics are deterministic;
//   float atomics would NOT be).
//
//   Memory: d_bases/d_offset/d_length read from global memory; d_counts updated
//   in global memory via atomics. No shared memory here -- a privatized per-block
//   histogram in shared memory is the classic optimization (see THEORY sec 4 +
//   the exercises), but the global-atomic version is the clearest to learn from.
// ---------------------------------------------------------------------------
__global__ void count_kmers_kernel(const char* __restrict__ d_bases,
                                   const int* __restrict__ d_offset,
                                   const int* __restrict__ d_length,
                                   int n,
                                   uint32_t* d_counts) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's read index
    if (g >= n) return;                              // guard ragged last block

    const char* seq = d_bases + d_offset[g];         // start of read g
    const int   len = d_length[g];                   // length of read g

    // Slide the window; kmer_code_at() (shared, from reference_cpu.h) returns the
    // code or 0xFFFFFFFF for a window that runs off the end or contains 'N'.
    for (int p = 0; p + KMER_K <= len; ++p) {
        uint32_t code = kmer_code_at(seq, len, p);
        if (code != 0xFFFFFFFFu) {
            // One hardware-atomic increment of this k-mer's slot. uint32 is wide
            // enough: even at deep coverage a single 9-mer count stays well below
            // 2^32. (Overflow would be a silent bug, hence the deliberate type.)
            atomicAdd(&d_counts[code], 1u);
        }
    }
}

// ---------------------------------------------------------------------------
// correct_reads_kernel  (PHASE 2: correct the reads)
//
//   Thread-to-data map: thread g owns READ g again. The spectrum is now FROZEN
//   (phase 1 finished and synchronized), so every thread only READS d_counts --
//   no atomics, no races. The thread calls the shared correct_one_read() physics,
//   which writes the corrected bytes into this read's slice of d_corrected and
//   returns the number of substitutions, stored in d_changes[g].
//
//   This is the "N independent jobs" pattern: read i's correction is completely
//   independent of read j's, so the work is embarrassingly parallel. Because the
//   thread runs the SAME inline function the CPU reference runs, byte-for-byte
//   agreement is guaranteed (verified in main.cu).
//
//   Memory: d_bases/d_offset/d_length and d_counts read from global memory;
//   d_corrected and d_changes written. correct_one_read writes the read into its
//   own output slice first (so reads never overlap) -- safe without atomics.
// ---------------------------------------------------------------------------
__global__ void correct_reads_kernel(const char* __restrict__ d_bases,
                                     const int* __restrict__ d_offset,
                                     const int* __restrict__ d_length,
                                     int n,
                                     const uint32_t* __restrict__ d_counts,
                                     uint32_t thresh,
                                     char* __restrict__ d_corrected,
                                     int* __restrict__ d_changes) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's read index
    if (g >= n) return;                              // guard ragged last block

    const char* in  = d_bases + d_offset[g];         // raw read g (read-only)
    char*       out = d_corrected + d_offset[g];     // corrected read g (output)
    const int   len = d_length[g];

    // The shared physics: identical to the CPU reference's per-read call.
    d_changes[g] = correct_one_read(in, out, len, d_counts, thresh);
}

// ---------------------------------------------------------------------------
// correct_reads_gpu: the host wrapper that runs BOTH phases.
//   Steps: (1) allocate + upload reads; (2) zero + build the spectrum (phase 1,
//   timed); (3) correct the reads (phase 2, timed); (4) download spectrum +
//   corrected reads + change counts; (5) free. We time each KERNEL with CUDA
//   events (not the copies) so the reported ms is the compute cost.
// ---------------------------------------------------------------------------
void correct_reads_gpu(const ReadSet& reads, uint32_t thresh,
                       std::vector<uint32_t>& counts_out,
                       std::vector<char>& corrected_out,
                       std::vector<int>& changes_out,
                       float* count_ms, float* correct_ms) {
    const int          n          = reads.n;
    const std::size_t  total      = reads.bases.size();              // all read bytes
    const std::size_t  base_bytes = total * sizeof(char);
    const std::size_t  off_bytes  = (static_cast<std::size_t>(n) + 1) * sizeof(int);
    const std::size_t  len_bytes  = static_cast<std::size_t>(n) * sizeof(int);
    const std::size_t  cnt_bytes  = static_cast<std::size_t>(KMER_TABLE_N) * sizeof(uint32_t);

    // Size the host outputs up front.
    counts_out.assign(KMER_TABLE_N, 0u);
    corrected_out.assign(total, 0);
    changes_out.assign(static_cast<std::size_t>(n), 0);

    // (1) Device buffers (d_ prefix = device pointer; never dereference on host).
    char*     d_bases     = nullptr;   // [total]        raw reads concatenated
    int*      d_offset    = nullptr;   // [n+1]          CSR offsets
    int*      d_length    = nullptr;   // [n]            per-read lengths
    uint32_t* d_counts    = nullptr;   // [KMER_TABLE_N] spectrum
    char*     d_corrected = nullptr;   // [total]        corrected reads
    int*      d_changes   = nullptr;   // [n]            per-read substitution counts
    CUDA_CHECK(cudaMalloc(&d_bases,     base_bytes));
    CUDA_CHECK(cudaMalloc(&d_offset,    off_bytes));
    CUDA_CHECK(cudaMalloc(&d_length,    len_bytes));
    CUDA_CHECK(cudaMalloc(&d_counts,    cnt_bytes));
    CUDA_CHECK(cudaMalloc(&d_corrected, base_bytes));
    CUDA_CHECK(cudaMalloc(&d_changes,   len_bytes));

    // Upload the read data H2D.
    CUDA_CHECK(cudaMemcpy(d_bases,  reads.bases.data(),  base_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offset, reads.offset.data(), off_bytes,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_length, reads.length.data(), len_bytes,  cudaMemcpyHostToDevice));

    const int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // (2) PHASE 1: zero the table, then count. cudaMemset on the device avoids a
    //     host round-trip; the counting kernel then atomicAdds into it.
    CUDA_CHECK(cudaMemset(d_counts, 0, cnt_bytes));
    {
        GpuTimer t;
        t.start();
        count_kmers_kernel<<<blocks, THREADS_PER_BLOCK>>>(
            d_bases, d_offset, d_length, n, d_counts);
        *count_ms = t.stop_ms();
        CUDA_CHECK_LAST("count_kmers_kernel");
    }

    // (3) PHASE 2: correct the reads against the (now frozen) spectrum.
    {
        GpuTimer t;
        t.start();
        correct_reads_kernel<<<blocks, THREADS_PER_BLOCK>>>(
            d_bases, d_offset, d_length, n, d_counts, thresh, d_corrected, d_changes);
        *correct_ms = t.stop_ms();
        CUDA_CHECK_LAST("correct_reads_kernel");
    }

    // (4) Download everything we want to verify/report.
    CUDA_CHECK(cudaMemcpy(counts_out.data(),    d_counts,    cnt_bytes,  cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(corrected_out.data(), d_corrected, base_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(changes_out.data(),   d_changes,   len_bytes,  cudaMemcpyDeviceToHost));

    // (5) Free (no GPU garbage collector exists).
    CUDA_CHECK(cudaFree(d_bases));
    CUDA_CHECK(cudaFree(d_offset));
    CUDA_CHECK(cudaFree(d_length));
    CUDA_CHECK(cudaFree(d_counts));
    CUDA_CHECK(cudaFree(d_corrected));
    CUDA_CHECK(cudaFree(d_changes));
}
