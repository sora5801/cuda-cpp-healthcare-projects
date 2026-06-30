// ===========================================================================
// src/kernels.cu  --  GPU k-mer counting (atomic hash table) + minimiser sketch
// ---------------------------------------------------------------------------
// Project 3.6 : k-mer Counting & Minimiser Sketching
//
// GPU twins of count_kmers_cpu() and sketch_cpu(). Both call the SAME per-k-mer
// math from kmer.h (encode/canonicalise/hash), so the GPU result matches the CPU
// result exactly. main.cu compares histograms key-by-key and sketches hash-by-
// hash. See ../THEORY.md "GPU mapping".
//
// The counting kernel implements a DEVICE OPEN-ADDRESSING HASH TABLE with linear
// probing, claiming slots with atomicCAS and tallying with atomicAdd -- a hand-
// rolled, teachable version of the lock-free tables in Gerbil/Jellyfish.
//
// READ THIS AFTER: kmer.h, kernels.cuh, util/cuda_check.cuh, util/timer.cuh.
// ===========================================================================
#include "kernels.cuh"
#include "kmer.h"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"

#include <algorithm>   // std::sort, std::unique
#include <cstdint>
#include <vector>

// A good occupancy default across sm_75..sm_89: 256 threads/block.
static constexpr int THREADS_PER_BLOCK = 256;

// ===========================================================================
// PART 1 -- k-MER COUNTING via a device open-addressing hash table
// ===========================================================================
//
// LAYOUT WE UPLOAD
//   d_bases  : the concatenated read characters (ReadSet::bases).
//   d_pos    : for each of the P global k-mer positions, the ABSOLUTE index into
//              d_bases where that k-mer starts. Precomputed on the host so a
//              thread does O(1) work (no per-thread binary search over offsets).
//   The table is two parallel device arrays of capacity C (a power of two):
//   d_keys[C] (canonical k-mer or KMER_EMPTY) and d_counts[C] (occurrences).
//
// WHY A HASH TABLE (and not sort-then-reduce)?
//   The catalog lists BOTH approaches. Sorting all k-mers then run-length-
//   encoding (thrust::sort_by_key) is the other classic route; we hand-roll the
//   hash table because it teaches the atomic-insert pattern directly and needs no
//   extra library. THEORY.md contrasts the two.
//
// DETERMINISM
//   Two threads inserting the SAME key race to claim the slot, but exactly one
//   wins the atomicCAS; both then atomicAdd 1 to the SAME counter. Integer adds
//   commute, so the final count is identical regardless of thread order. The SET
//   of (key,count) pairs is therefore order-independent; the host sorts it by key
//   for a byte-stable printout. (PATTERNS.md section 3.)

// ---------------------------------------------------------------------------
// hash_insert_kernel: one thread inserts one k-mer into the device hash table.
//   grid  : ceil(P / 256) blocks ; block : 256 threads
//   thread global id `t` -> k-mer that starts at d_bases[d_pos[t]]
//
//   d_bases   : [total_bases] concatenated read chars
//   d_pos     : [P] absolute start index of each candidate k-mer
//   P         : number of candidate k-mer positions
//   k         : k-mer length
//   d_keys    : [cap] table keys, pre-filled with KMER_EMPTY
//   d_counts  : [cap] table counts, pre-zeroed
//   cap_mask  : cap-1 (cap is a power of two => index & cap_mask == index % cap)
//
//   Memory spaces: all global memory; atomics on global. No shared memory (the
//   table is far larger than a block could hold). __restrict__ promises no
//   aliasing so loads can be cached in registers.
// ---------------------------------------------------------------------------
__global__ void hash_insert_kernel(const char* __restrict__ d_bases,
                                   const std::size_t* __restrict__ d_pos,
                                   int P, int k,
                                   uint64_t* __restrict__ d_keys,
                                   unsigned int* __restrict__ d_counts,
                                   unsigned int cap_mask) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= P) return;                                  // guard the ragged last block

    // Encode + canonicalise this thread's k-mer. If the window has an invalid
    // base (e.g. 'N'), skip it -- exactly like the CPU reference.
    uint64_t canon, hash;
    if (!canonical_hash_at(d_bases, d_pos[t], k, &canon, &hash)) return;

    // LINEAR PROBING. Start at slot (hash & cap_mask); on collision step +1
    // (wrapping) until we either (a) find our key already present, or (b) claim an
    // empty slot for it. We loop at most `cap` times (the table is sized with
    // headroom in the host wrapper so it never fills -- a load factor < 0.5).
    unsigned int slot = (unsigned int)(hash) & cap_mask;
    for (unsigned int probe = 0; probe <= cap_mask; ++probe) {
        uint64_t cur = d_keys[slot];
        if (cur == canon) {                              // key already here
            atomicAdd(&d_counts[slot], 1u);              // tally (integer => commutes)
            return;
        }
        if (cur == KMER_EMPTY) {
            // Try to CLAIM this empty slot for `canon`. atomicCAS returns the
            // value that was there BEFORE our attempt:
            //   * KMER_EMPTY  -> we won; the slot is now ours.
            //   * == canon    -> someone else inserted the SAME key first; share it.
            //   * other key   -> someone took it for a different key; keep probing.
            uint64_t prev = atomicCAS((unsigned long long*)&d_keys[slot],
                                      (unsigned long long)KMER_EMPTY,
                                      (unsigned long long)canon);
            if (prev == KMER_EMPTY || prev == canon) {
                atomicAdd(&d_counts[slot], 1u);
                return;
            }
            // else: lost the race to a DIFFERENT key -> fall through and probe on.
        }
        slot = (slot + 1u) & cap_mask;                   // next slot (wrap via mask)
    }
    // Unreachable if the table has headroom (host guarantees load factor < 0.5).
}

// ---------------------------------------------------------------------------
// count_kmers_gpu: host wrapper. Builds the position map, runs the insert kernel,
//   compacts the table to (key,count) pairs, and sorts ascending by key.
// ---------------------------------------------------------------------------
std::vector<KmerCount> count_kmers_gpu(const ReadSet& rs, float* kernel_ms) {
    const int k = rs.k;

    // --- 1. Build the per-position map on the host -------------------------
    // For every read, every valid k-mer START position contributes one entry:
    // the ABSOLUTE index into rs.bases. (We include positions whose window may
    // contain an 'N'; the kernel skips those, matching the CPU which also slides
    // over every position and skips invalid windows.)
    std::vector<std::size_t> pos;
    for (int r = 0; r < rs.num_reads; ++r) {
        const std::size_t start = rs.offsets[r];
        const std::size_t len   = rs.read_len(r);
        if (len < (std::size_t)k) continue;
        const std::size_t n_windows = len - k + 1;
        for (std::size_t p = 0; p < n_windows; ++p) pos.push_back(start + p);
    }
    const int P = (int)pos.size();

    // --- 2. Size the hash table: next power of two >= 2*P (load factor < 0.5)--
    // Headroom keeps linear-probe chains short and guarantees the kernel's probe
    // loop always finds a slot. We also need at least a few slots if P is tiny.
    std::size_t cap = 16;
    while (cap < (std::size_t)P * 2) cap <<= 1;           // grow to a power of two
    const unsigned int cap_mask = (unsigned int)(cap - 1);

    // --- 3. Device buffers + uploads --------------------------------------
    char* d_bases = nullptr; std::size_t* d_pos = nullptr;
    uint64_t* d_keys = nullptr; unsigned int* d_counts = nullptr;
    CUDA_CHECK(cudaMalloc(&d_bases, rs.bases.size() * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&d_pos,   (std::size_t)P * sizeof(std::size_t)));
    CUDA_CHECK(cudaMalloc(&d_keys,  cap * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_counts, cap * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(d_bases, rs.bases.data(), rs.bases.size() * sizeof(char),
                          cudaMemcpyHostToDevice));
    if (P > 0)
        CUDA_CHECK(cudaMemcpy(d_pos, pos.data(), (std::size_t)P * sizeof(std::size_t),
                              cudaMemcpyHostToDevice));

    // Pre-fill keys with KMER_EMPTY (0xFF bytes == all-ones) and zero the counts.
    CUDA_CHECK(cudaMemset(d_keys, 0xFF, cap * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_counts, 0, cap * sizeof(unsigned int)));

    // --- 4. Launch the insert kernel (timed with CUDA events) -------------
    GpuTimer timer;
    timer.start();
    if (P > 0) {
        const int grid = (P + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        hash_insert_kernel<<<grid, THREADS_PER_BLOCK>>>(d_bases, d_pos, P, k,
                                                        d_keys, d_counts, cap_mask);
    }
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("hash_insert_kernel");

    // --- 5. Bring the table back; compact non-empty slots -----------------
    std::vector<uint64_t>     h_keys(cap);
    std::vector<unsigned int> h_counts(cap);
    CUDA_CHECK(cudaMemcpy(h_keys.data(),   d_keys,   cap * sizeof(uint64_t),     cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_counts.data(), d_counts, cap * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_bases));
    CUDA_CHECK(cudaFree(d_pos));
    CUDA_CHECK(cudaFree(d_keys));
    CUDA_CHECK(cudaFree(d_counts));

    std::vector<KmerCount> out;
    for (std::size_t i = 0; i < cap; ++i)
        if (h_keys[i] != KMER_EMPTY) out.push_back({h_keys[i], h_counts[i]});

    // --- 6. Sort ascending by key for a DETERMINISTIC printout ------------
    std::sort(out.begin(), out.end(),
              [](const KmerCount& a, const KmerCount& b) { return a.key < b.key; });
    return out;
}

// ===========================================================================
// PART 2 -- MINIMISER SKETCHING
// ===========================================================================
//
// We give one thread to each MINIMISER WINDOW. A minimiser window is w
// consecutive k-mers; the thread scans their hashes and writes the minimum. We
// precompute, on the host, a per-window map: the absolute base index of the
// window's FIRST k-mer, and a flag for whether the read is long enough. The
// kernel re-hashes each of the w k-mers (cheap, and keeps the device code's math
// identical to the CPU's). The host then sorts/dedups/truncates to bottom-s.
//
// A production kernel would compute window minima with warp shuffles
// (__shfl_down_sync) so a warp cooperatively reduces 32 lanes; we keep the
// explicit per-window loop because it is far easier to read and the result is
// identical. THEORY.md sketches the warp version.

// ---------------------------------------------------------------------------
// minimiser_kernel: one thread emits one window's minimum k-mer hash.
//   d_bases   : [total_bases] concatenated read chars
//   d_win     : [W] absolute base index of each window's first k-mer
//   W         : number of minimiser windows across all reads
//   k, w      : k-mer length and window length (in k-mers)
//   d_out     : [W] output minimum hash per window (KMER_EMPTY if all invalid)
// ---------------------------------------------------------------------------
__global__ void minimiser_kernel(const char* __restrict__ d_bases,
                                 const std::size_t* __restrict__ d_win,
                                 int W, int k, int w,
                                 uint64_t* __restrict__ d_out) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= W) return;

    const std::size_t base0 = d_win[t];                  // window's first k-mer start
    uint64_t best = KMER_EMPTY;                          // running window minimum
    // Scan the w k-mers of this window; canonical hash of each; keep the min.
    for (int j = 0; j < w; ++j) {
        uint64_t canon, hash;
        if (canonical_hash_at(d_bases, base0 + j, k, &canon, &hash))
            best = (hash < best) ? hash : best;
    }
    d_out[t] = best;                                     // KMER_EMPTY => no valid k-mer
}

// ---------------------------------------------------------------------------
// sketch_gpu: host wrapper. Builds the window map, runs the minimiser kernel,
//   then sorts/dedups/truncates to the bottom-s sketch (same as the CPU).
// ---------------------------------------------------------------------------
Sketch sketch_gpu(const ReadSet& rs, int s, float* kernel_ms) {
    const int k = rs.k;
    const int w = rs.w;

    // --- 1. Build the per-window map on the host --------------------------
    // For each read with at least w k-mers, every minimiser-window start
    // contributes the absolute base index of the window's first k-mer.
    std::vector<std::size_t> win;
    for (int r = 0; r < rs.num_reads; ++r) {
        const std::size_t start = rs.offsets[r];
        const std::size_t len   = rs.read_len(r);
        if (len < (std::size_t)k) continue;
        const std::size_t n_windows = len - k + 1;       // # k-mers in this read
        if (n_windows < (std::size_t)w) continue;        // too short for a full window
        const std::size_t n_min_windows = n_windows - w + 1;
        for (std::size_t i = 0; i < n_min_windows; ++i) win.push_back(start + i);
    }
    const int W = (int)win.size();

    // --- 2. Device buffers + uploads --------------------------------------
    char* d_bases = nullptr; std::size_t* d_win = nullptr; uint64_t* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_bases, rs.bases.size() * sizeof(char)));
    CUDA_CHECK(cudaMemcpy(d_bases, rs.bases.data(), rs.bases.size() * sizeof(char),
                          cudaMemcpyHostToDevice));
    // Allocate at least one element so cudaMalloc(0) is never an issue.
    CUDA_CHECK(cudaMalloc(&d_win, (std::size_t)(W > 0 ? W : 1) * sizeof(std::size_t)));
    CUDA_CHECK(cudaMalloc(&d_out, (std::size_t)(W > 0 ? W : 1) * sizeof(uint64_t)));
    if (W > 0)
        CUDA_CHECK(cudaMemcpy(d_win, win.data(), (std::size_t)W * sizeof(std::size_t),
                              cudaMemcpyHostToDevice));

    // --- 3. Launch the minimiser kernel (timed) ---------------------------
    GpuTimer timer;
    timer.start();
    if (W > 0) {
        const int grid = (W + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        minimiser_kernel<<<grid, THREADS_PER_BLOCK>>>(d_bases, d_win, W, k, w, d_out);
    }
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("minimiser_kernel");

    // --- 4. Pull window minima back; bottom-s on the host -----------------
    std::vector<uint64_t> mins(W > 0 ? W : 0);
    if (W > 0)
        CUDA_CHECK(cudaMemcpy(mins.data(), d_out, (std::size_t)W * sizeof(uint64_t),
                              cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_bases));
    CUDA_CHECK(cudaFree(d_win));
    CUDA_CHECK(cudaFree(d_out));

    // Drop "no valid k-mer" sentinels, then sort/dedup/truncate to bottom-s.
    std::vector<uint64_t> valid;
    valid.reserve(mins.size());
    for (uint64_t h : mins) if (h != KMER_EMPTY) valid.push_back(h);
    std::sort(valid.begin(), valid.end());
    valid.erase(std::unique(valid.begin(), valid.end()), valid.end());
    if ((int)valid.size() > s) valid.resize(s);

    Sketch out;
    out.hashes = std::move(valid);
    return out;
}
