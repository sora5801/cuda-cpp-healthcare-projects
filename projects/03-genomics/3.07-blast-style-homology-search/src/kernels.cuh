// ===========================================================================
// src/kernels.cuh  --  GPU compute interface for BLAST-style homology search
// ---------------------------------------------------------------------------
// Project 3.7 : BLAST-Style Homology Search
//
// THE BIG IDEA  (PATTERNS.md sec 1, row "score one query vs N items, each
// independent" -- the same family as flagship 1.12 Tanimoto)
//   Searching the query against N database sequences is N INDEPENDENT jobs, so
//   we give each DB sequence its OWN GPU thread. Each thread:
//     1. slides a length-k window over its sequence,
//     2. looks up each k-mer in the QUERY index (a sorted (code,qpos) array in
//        global memory, binary-searched on device),
//     3. for every seed hit, runs gapless X-drop extension (blast_core.h,
//        shared verbatim with the CPU), and
//     4. keeps the best HSP score, writing one integer to out[i].
//
//   Two CUDA features carry the teaching weight, mirroring 1.12:
//     * the BLOSUM62 matrix (576 bytes) lives in CONSTANT memory -- every thread
//       reads it, none writes it, so the constant cache broadcasts it warp-wide;
//     * a grid-stride loop lets one modest grid cover an arbitrarily large DB.
//   Because all scoring is INTEGER, the GPU result equals the CPU result EXACTLY
//   (verify tolerance 0, PATTERNS.md sec 4).
//
//   This header is included only by .cu units. main.cu calls blast_gpu().
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, reference_cpu.h,
// blast_core.h. Then read kernels.cu. The GPU mapping is in ../THEORY.md.
// ===========================================================================
#pragma once

#include <cstdint>
#include <vector>

#include "reference_cpu.h"   // SequenceDB, QueryIndex, SEED_K, X_DROP (pure C++)

// ---------------------------------------------------------------------------
// SeedPair: one (k-mer code, query position) entry of the FLATTENED query index.
//   The host hash map (QueryIndex) is great for the CPU but unfriendly to a GPU
//   thread. So we flatten it to a single array of SeedPairs SORTED by `code`;
//   a device thread then BINARY-SEARCHES this array for a DB k-mer's code and
//   walks the contiguous run of equal codes to recover every query position.
//   This is the GPU analogue of BLAST's lookup table.
// ---------------------------------------------------------------------------
struct SeedPair {
    int code;   // packed base-24 k-mer code (see pack_kmer in reference_cpu.h)
    int qpos;   // a query position where that k-mer occurs
};

// flatten_query_index: turn the host QueryIndex hash map into a sorted SeedPair
//   array the kernel can binary-search. Defined in kernels.cu (host-side helper).
//   Sorting by (code, qpos) makes the search deterministic AND keeps all query
//   positions for a given code contiguous.
std::vector<SeedPair> flatten_query_index(const QueryIndex& qi);

// ---------------------------------------------------------------------------
// blast_kernel: one thread per DB sequence. Reads the query (encoded) and the
// concatenated DB from global memory, the sorted seed array from global memory,
// and BLOSUM62 from constant memory; writes out[i] = best HSP score of DB seq i.
//   d_query / query_len : the encoded query residues.
//   d_db_res            : all DB residues, concatenated (coalesced array).
//   d_db_off / d_db_len : [n] offset and length of each DB sequence.
//   d_seeds / n_seeds   : the sorted flattened query index.
//   n                   : number of DB sequences.
//   out                 : [n] best HSP score per DB sequence (output).
// ---------------------------------------------------------------------------
__global__ void blast_kernel(const int8_t* __restrict__ d_query, int query_len,
                             const int8_t* __restrict__ d_db_res,
                             const int*    __restrict__ d_db_off,
                             const int*    __restrict__ d_db_len,
                             const SeedPair* __restrict__ d_seeds, int n_seeds,
                             int n, int* __restrict__ out);

// ---------------------------------------------------------------------------
// blast_gpu: host wrapper. Uploads the query, the DB, the flattened seed array,
// and BLOSUM62 (to constant memory), launches the kernel, times ONLY the kernel
// (CUDA events), and returns one best-HSP score per DB sequence.
//   db        : the loaded dataset (query + DB sequences).
//   query_idx : the prebuilt query k-mer index (flattened here for the device).
//   out       : resized to db.n; filled with best HSP score per DB sequence.
//   kernel_ms : out-param, GPU-measured kernel time in milliseconds.
// ---------------------------------------------------------------------------
void blast_gpu(const SequenceDB& db, const QueryIndex& query_idx,
               std::vector<int>& out, float* kernel_ms);
