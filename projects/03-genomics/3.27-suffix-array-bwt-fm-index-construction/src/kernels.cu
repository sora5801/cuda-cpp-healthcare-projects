// ===========================================================================
// src/kernels.cu  --  GPU prefix-doubling suffix array (radix-sort core)
// ---------------------------------------------------------------------------
// Project 3.27 : Suffix Array / BWT / FM-Index Construction
//
// WHAT THIS FILE DOES
//   Implements the GPU twin of suffix_array_cpu(): it builds the suffix array by
//   PREFIX DOUBLING, where each round sorts all n suffixes by a packed 64-bit
//   rank-pair key. The sort is a hand-rolled, deterministic LSD RADIX SORT (8
//   passes of 8-bit digits) -- the exact primitive thrust::sort_by_key would use
//   under the hood, written out so nothing is a black box (CLAUDE.md section 6.1.6).
//
//   The per-suffix key math is shared with the CPU via sa_core.h (pack_key), so
//   the GPU and CPU produce BIT-IDENTICAL suffix arrays -> exact verification.
//
//   Pipeline per doubling round (k = 1, 2, 4, ...):
//     build_keys_kernel  -> key[i] = pack_key(i, k, n, rank)
//     [ radix sort (key, val=suffixIndex) by 8x 8-bit digits ]
//     flag_kernel        -> flag[p] = (sortedKey[p] != sortedKey[p-1])
//     exclusive scan     -> prefix[p] = sum of flags before p  (= new rank)
//     write_ranks_kernel -> rank[val[p]] = prefix[p]
//   Stop when the number of distinct ranks reaches n (every suffix unique).
//
//   ALL accumulation that crosses threads uses INTEGER atomics / integer scans,
//   never floating point -> the result is deterministic and reproducible
//   (PATTERNS.md section 3). The reported result on stdout never varies.
//
// READ THIS AFTER: kernels.cuh (declarations), sa_core.h (the shared key math).
// ===========================================================================
#include "kernels.cuh"
#include "sa_core.h"             // pack_key (shared host/device key math)
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer

#include <cstdint>
#include <vector>

// 256 threads/block: a multiple of the 32-lane warp, good occupancy on sm_75..89.
static constexpr int THREADS_PER_BLOCK = 256;
// Radix sort parameters: 8-bit digits => 256 buckets, 8 passes cover 64 bits.
static constexpr int RADIX_BITS    = 8;
static constexpr int RADIX_BUCKETS = 1 << RADIX_BITS;   // 256
static constexpr int RADIX_PASSES  = 64 / RADIX_BITS;   // 8

// ===========================================================================
// 1. KEY CONSTRUCTION
// ===========================================================================
// build_keys_kernel: one thread per SORTED SLOT packs the rank pair of the
//   suffix currently at that slot (val[p]) into key[p], using the SHARED
//   pack_key() (so the CPU and GPU sort by identical keys). We key the
//   MAINTAINED order -- not the identity -- so that the stable radix sort
//   carries the prior round's order as the tie-break and converges to the unique
//   suffix array, exactly as the CPU reference does.
//   grid = ceil(n / 256), block = 256; thread p owns sorted slot p.
__global__ void build_keys_kernel(int n, int k, const int* __restrict__ val,
                                  const int* __restrict__ rank,
                                  std::uint64_t* __restrict__ key) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;  // this thread's slot
    if (p >= n) return;                                   // guard ragged last block
    key[p] = pack_key(val[p], k, n, rank);                // (rank[s], rank[s+k]) -> u64
}

// ===========================================================================
// 2. LSD RADIX SORT  (stable, deterministic)
// ===========================================================================
// histogram_kernel: for one 8-bit digit position (given by `shift`), count how
//   many keys fall in each of the 256 buckets. Every thread extracts its key's
//   digit and atomicAdds 1 into that bucket. Integer atomicAdd is associative,
//   so the final histogram is identical no matter the thread order -> deterministic.
__global__ void histogram_kernel(int n, int shift, const std::uint64_t* __restrict__ key_in,
                                 unsigned int* __restrict__ hist) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // (key >> shift) & 0xFF selects this pass's 8-bit digit.
    const unsigned int digit = static_cast<unsigned int>((key_in[i] >> shift) & 0xFFu);
    atomicAdd(&hist[digit], 1u);   // bump that bucket's count (integer -> exact)
}

// scatter_kernel: place every record into its final sorted slot FOR THIS PASS.
//   A correct LSD radix sort must be STABLE (equal digits keep their input
//   order). The simplest provably-stable scatter is a single sequential walk
//   that, for each record in input order, claims the next free slot of its
//   bucket (offset[digit]++). We therefore launch this kernel with ONE thread:
//   clarity and guaranteed determinism over raw speed (the radix passes are
//   tiny for teaching sizes; THEORY discusses the parallel-stable-scatter that a
//   production sort uses). `offset` arrives as the EXCLUSIVE prefix sum of the
//   histogram (the first output index of each bucket) and is mutated in place.
__global__ void scatter_kernel(int n, int shift,
                               const std::uint64_t* __restrict__ key_in,
                               const int* __restrict__ val_in,
                               std::uint64_t* __restrict__ key_out,
                               int* __restrict__ val_out,
                               unsigned int* __restrict__ offset) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;   // single-threaded by design
    for (int i = 0; i < n; ++i) {
        const unsigned int digit = static_cast<unsigned int>((key_in[i] >> shift) & 0xFFu);
        const unsigned int pos = offset[digit]++;       // next free slot of this bucket
        key_out[pos] = key_in[i];                       // move key  ...
        val_out[pos] = val_in[i];                       // ... and its suffix index together
    }
}

// ===========================================================================
// 3. RANK RENUMBERING  (flag -> exclusive scan -> scatter)
// ===========================================================================
// flag_kernel: mark a "1" wherever the sorted key changes from the previous slot.
//   flag[0] = 0 by definition. The exclusive prefix sum of flag[] is exactly the
//   new rank of the suffix sitting at each sorted slot (ranks start at 0 and
//   increase by 1 at every key boundary) -- identical to the CPU's renumber().
__global__ void flag_kernel(int n, const std::uint64_t* __restrict__ sorted_key,
                            int* __restrict__ flag) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;
    flag[p] = (p == 0) ? 0 : (sorted_key[p] != sorted_key[p - 1] ? 1 : 0);
}

// write_ranks_kernel: scatter the new rank to the suffix index it belongs to.
//   At sorted slot p sits suffix val[p] with new rank prefix[p]; write it home.
__global__ void write_ranks_kernel(int n, const int* __restrict__ val,
                                   const int* __restrict__ prefix,
                                   int* __restrict__ rank) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;
    rank[val[p]] = prefix[p];
}

// scan_inclusive_kernel: a single-block INCLUSIVE prefix sum over an int array
//   of length n (n small for teaching). out[p] = sum(in[0..p]). With the flag
//   array (flag[0]=0, flag[q]=1 at a key boundary) this yields EXACTLY the new
//   rank of the suffix at sorted slot p: the count of key boundaries at-or-before
//   p -- matching the CPU's renumber() (rank starts at 0 and bumps at each
//   boundary). NOTE: it must be inclusive, not exclusive; an exclusive scan would
//   drop the boundary AT p and give every tied group the wrong rank.
//   We run it single-threaded so the scan is sequential and deterministic; for
//   large n a multi-block Blelloch scan (or CUB) would replace this (see THEORY).
//   out_total[0] receives the grand total (= number of key boundaries), so the
//   host computes distinct ranks = out_total + 1 without a separate reduction.
__global__ void scan_inclusive_kernel(int n, const int* __restrict__ in,
                                      int* __restrict__ out, int* __restrict__ out_total) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;   // single-threaded scan
    int running = 0;
    for (int i = 0; i < n; ++i) {
        running += in[i];       // inclusive: add in[i] BEFORE storing
        out[i] = running;       // out[i] = sum(in[0..i]) = rank of slot i
    }
    *out_total = running;       // total boundaries; distinct ranks = running + 1
}

// ===========================================================================
// HOST WRAPPER: orchestrate the doubling rounds on the GPU.
// ===========================================================================
SaResult suffix_array_gpu(const std::string& text, const std::string& pattern, float* kernel_ms) {
    const int n = static_cast<int>(text.size());
    SaResult res;
    res.n = n;

    // ---- Device buffers ---------------------------------------------------
    // Two ping-pong copies of (key, val) for the radix passes, plus rank, flags,
    // prefix, and the small 256-bucket histogram/offset arrays.
    std::uint64_t *d_keyA = nullptr, *d_keyB = nullptr;
    int  *d_valA = nullptr, *d_valB = nullptr;
    int  *d_rank = nullptr, *d_flag = nullptr, *d_prefix = nullptr, *d_total = nullptr;
    unsigned int *d_hist = nullptr, *d_offset = nullptr;

    const std::size_t nbytes_u64 = static_cast<std::size_t>(n) * sizeof(std::uint64_t);
    const std::size_t nbytes_i32 = static_cast<std::size_t>(n) * sizeof(int);
    CUDA_CHECK(cudaMalloc(&d_keyA, nbytes_u64));
    CUDA_CHECK(cudaMalloc(&d_keyB, nbytes_u64));
    CUDA_CHECK(cudaMalloc(&d_valA, nbytes_i32));
    CUDA_CHECK(cudaMalloc(&d_valB, nbytes_i32));
    CUDA_CHECK(cudaMalloc(&d_rank, nbytes_i32));
    CUDA_CHECK(cudaMalloc(&d_flag, nbytes_i32));
    CUDA_CHECK(cudaMalloc(&d_prefix, nbytes_i32));
    CUDA_CHECK(cudaMalloc(&d_total, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_hist, RADIX_BUCKETS * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_offset, RADIX_BUCKETS * sizeof(unsigned int)));

    // ---- Initial ranks (k = 0): single-character codes --------------------
    // We compute the initial ranks and identity values on the host and upload
    // them; this is O(n) and one-time, so it stays off the timed kernel path.
    std::vector<int> h_rank(n), h_val(n);
    for (int i = 0; i < n; ++i) {
        h_rank[i] = char_to_code(text[i]);   // shared sa_core.h codes
        h_val[i]  = i;                        // identity suffix order to start
    }
    CUDA_CHECK(cudaMemcpy(d_rank, h_rank.data(), nbytes_i32, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_valA, h_val.data(), nbytes_i32, cudaMemcpyHostToDevice));

    const int grid = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // We time the whole doubling loop with CUDA events (the GPU "compute" cost).
    GpuTimer timer;
    timer.start();

    int rounds = 0;
    std::vector<unsigned int> h_hist(RADIX_BUCKETS), h_offset(RADIX_BUCKETS);

    // d_valA holds the MAINTAINED suffix order (the running permutation). It
    // starts as the identity (uploaded above) and each round is re-sorted in
    // place by the new keys. Carrying it across rounds preserves the tie-break
    // that makes prefix doubling converge to the unique suffix array.
    for (int k = 1; k < n; k <<= 1) {
        // (a) Build this round's key for each sorted SLOT from the suffix that
        //     currently sits there (d_valA[p]). Keying the maintained order --
        //     not the identity -- is what makes the stable sort reproduce the
        //     CPU's exact suffix array (see build_keys_kernel + THEORY).
        build_keys_kernel<<<grid, THREADS_PER_BLOCK>>>(n, k, d_valA, d_rank, d_keyA);
        CUDA_CHECK_LAST("build_keys_kernel");

        // (c) LSD radix sort of (d_keyA,d_valA) -> sorted, ping-ponging A<->B.
        std::uint64_t* keyIn = d_keyA; std::uint64_t* keyOut = d_keyB;
        int*           valIn = d_valA; int*           valOut = d_valB;
        for (int pass = 0; pass < RADIX_PASSES; ++pass) {
            const int shift = pass * RADIX_BITS;
            // Histogram of this pass's digit.
            CUDA_CHECK(cudaMemset(d_hist, 0, RADIX_BUCKETS * sizeof(unsigned int)));
            histogram_kernel<<<grid, THREADS_PER_BLOCK>>>(n, shift, keyIn, d_hist);
            CUDA_CHECK_LAST("histogram_kernel");
            // Exclusive prefix sum of the 256 buckets -> per-bucket start offset.
            // Tiny (256 ints); do it on the host for clarity (off the hot path
            // conceptually, though inside the timed region -- negligible).
            CUDA_CHECK(cudaMemcpy(h_hist.data(), d_hist,
                                  RADIX_BUCKETS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
            unsigned int run = 0;
            for (int b = 0; b < RADIX_BUCKETS; ++b) { h_offset[b] = run; run += h_hist[b]; }
            CUDA_CHECK(cudaMemcpy(d_offset, h_offset.data(),
                                  RADIX_BUCKETS * sizeof(unsigned int), cudaMemcpyHostToDevice));
            // Stable scatter into the sorted slots for this digit.
            scatter_kernel<<<1, 1>>>(n, shift, keyIn, valIn, keyOut, valOut, d_offset);
            CUDA_CHECK_LAST("scatter_kernel");
            // Swap in<->out for the next pass.
            std::swap(keyIn, keyOut);
            std::swap(valIn, valOut);
        }
        // After 8 passes the SORTED arrays are back in (keyIn, valIn) (even #passes).

        // (d) Renumber ranks: flag key changes, exclusive-scan, scatter ranks.
        flag_kernel<<<grid, THREADS_PER_BLOCK>>>(n, keyIn, d_flag);
        CUDA_CHECK_LAST("flag_kernel");
        scan_inclusive_kernel<<<1, 1>>>(n, d_flag, d_prefix, d_total);
        CUDA_CHECK_LAST("scan_inclusive_kernel");
        write_ranks_kernel<<<grid, THREADS_PER_BLOCK>>>(n, valIn, d_prefix, d_rank);
        CUDA_CHECK_LAST("write_ranks_kernel");

        ++rounds;

        // distinct ranks = (largest rank) + 1. The inclusive scan's grand total
        // d_total is the number of key boundaries = the largest rank, so
        // distinct = d_total + 1. When this hits n every suffix is unique.
        int total_flags = 0;
        CUDA_CHECK(cudaMemcpy(&total_flags, d_total, sizeof(int), cudaMemcpyDeviceToHost));
        const int distinct = total_flags + 1;

        // Keep the current sorted suffix order (valIn) as the candidate SA; if we
        // are done we copy it out below.
        if (distinct == n) {
            // Copy the final suffix array (valIn) out of the device.
            res.sa.resize(n);
            CUDA_CHECK(cudaMemcpy(res.sa.data(), valIn, nbytes_i32, cudaMemcpyDeviceToHost));
            break;
        }
        // Otherwise loop again with the updated ranks in d_rank. We always need
        // the final sorted order; if the loop exits WITHOUT hitting distinct==n
        // (only possible when k overflows before uniqueness, which cannot happen
        // for a sentineled string), we still copy below as a safety net.
        if ((k << 1) >= n) {
            res.sa.resize(n);
            CUDA_CHECK(cudaMemcpy(res.sa.data(), valIn, nbytes_i32, cudaMemcpyDeviceToHost));
        }
    }

    *kernel_ms = timer.stop_ms();
    res.doubling_rounds = rounds;

    // Safety: if the text was length 1 (just "$"), the loop never ran; SA = {0}.
    if (res.sa.empty()) {
        res.sa.assign(n, 0);
        for (int i = 0; i < n; ++i) res.sa[i] = i;
    }

    // ---- Free device memory ----------------------------------------------
    CUDA_CHECK(cudaFree(d_keyA));   CUDA_CHECK(cudaFree(d_keyB));
    CUDA_CHECK(cudaFree(d_valA));   CUDA_CHECK(cudaFree(d_valB));
    CUDA_CHECK(cudaFree(d_rank));   CUDA_CHECK(cudaFree(d_flag));
    CUDA_CHECK(cudaFree(d_prefix)); CUDA_CHECK(cudaFree(d_total));
    CUDA_CHECK(cudaFree(d_hist));   CUDA_CHECK(cudaFree(d_offset));

    // ---- Host-side postprocessing (shared with the CPU path) --------------
    // Derive BWT and FM count from the GPU's SA using the SAME helpers the CPU
    // uses, so these can never disagree given an identical SA.
    res.bwt = bwt_from_sa(text, res.sa);
    res.pattern_count = fm_count(text, res.sa, pattern);
    return res;
}
