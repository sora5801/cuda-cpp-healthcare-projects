// ===========================================================================
// src/kernels.cu  --  GPU sort + dedup (Thrust radix sort + segmented reduction)
// ---------------------------------------------------------------------------
// Project 3.26 : GPU BAM Sorting & Deduplication
//
// GPU twin of reference_cpu.cpp. Both call the SAME integer key/compare helpers
// from bam.h, so the GPU result is byte-identical to the CPU baseline. main.cu
// runs both and verifies equality EXACTLY (no epsilon -- all integers). See
// ../THEORY.md "GPU mapping".
//
// The two device kernels here are tiny MAP kernels (one thread per read). The
// heavy lifting -- the radix sort and the segmented reduction -- is done by
// Thrust, which we explain inline (PATTERNS.md §5: use the library, never as a
// black box).
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"

// Thrust is header-only and ships with the CUDA toolkit, so these includes add
// NO extra .lib dependency. Each header brings one primitive we use below.
#include <thrust/device_vector.h>     // thrust::device_vector (GPU array)
#include <thrust/host_vector.h>       // thrust::host_vector (host staging)
#include <thrust/sort.h>              // thrust::sort_by_key / stable_sort_by_key
#include <thrust/reduce.h>            // thrust::reduce_by_key (segmented reduction)
#include <thrust/scan.h>             // thrust::inclusive_scan (group ids)
#include <thrust/transform.h>        // thrust::transform (boundary flags)
#include <cuda/std/functional>       // cuda::std::equal_to / not_equal_to (predicates)

#include <cstdint>

// One block of 256 threads is a solid occupancy default on sm_75..sm_89 for the
// trivial map kernels below (they are memory-bound, so block size is not
// critical; 256 keeps enough warps resident to hide global-memory latency).
static constexpr int THREADS_PER_BLOCK = 256;

// ===========================================================================
// (1) COORDINATE SORT
// ===========================================================================

// Map kernel: compute each read's packed coordinate key.
//   grid   : ceil(n / 256) blocks ;  block : 256 threads
//   thread (blockIdx.x, threadIdx.x) -> read index i = bx*blockDim.x + tx
//   Reads ReadRecord i from global memory, writes its uint64 coord_key and its
//   original id into parallel output arrays (a "structure-of-arrays" split that
//   Thrust's sort_by_key consumes). No atomics, no shared memory -- pure map.
__global__ void coord_key_kernel(const ReadRecord* __restrict__ reads, int n,
                                 uint64_t* __restrict__ keys,
                                 int* __restrict__ ids) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;                  // guard the ragged last block
    keys[i] = coord_key(reads[i]);       // bam.h: (ref<<40)|(pos<<16)|strand
    ids[i]  = reads[i].id;               // carry the id so we can tie-break + gather
}

// sort_gpu: see kernels.cuh. Strategy to get a TOTAL order that matches the CPU:
//   A radix sort_by_key on coord_key alone would leave ties among reads sharing
//   (ref,pos,strand) in an arbitrary order. We make the order total by sorting
//   TWICE, least-significant key first (the classic "stable radix on each field"
//   composition):
//     pass A: sort by id ascending            -> records ordered by id
//     pass B: STABLE-sort by coord_key         -> equal coord_keys keep id order
//   The result equals coord_less (coord_key, then id) exactly. We then GATHER
//   the full ReadRecords into genome order on the host using the sorted ids
//   (a tiny, readable permutation; the heavy sort ran on the GPU).
void sort_gpu(const ReadSet& rs, std::vector<ReadRecord>& out, float* kernel_ms) {
    const int n = rs.n();
    out.resize(static_cast<std::size_t>(n));

    // Upload the read records once.
    thrust::device_vector<ReadRecord> d_reads(rs.reads.begin(), rs.reads.end());
    thrust::device_vector<uint64_t>   d_keys(n);
    thrust::device_vector<int>        d_ids(n);

    const int grid = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    GpuTimer timer;
    timer.start();

    // MAP: fill (keys, ids) from the records.
    coord_key_kernel<<<grid, THREADS_PER_BLOCK>>>(
        thrust::raw_pointer_cast(d_reads.data()), n,
        thrust::raw_pointer_cast(d_keys.data()),
        thrust::raw_pointer_cast(d_ids.data()));
    CUDA_CHECK_LAST("coord_key_kernel");

    // PASS A -- order by id ascending (the tie-break field). After this, equal
    // coord_keys will appear in id order, which PASS B preserves (stable).
    //   sort_by_key(keys=ids, values=keys): sorts the id array and carries the
    //   coord_key array along so the two stay paired.
    thrust::sort_by_key(d_ids.begin(), d_ids.end(), d_keys.begin());

    // PASS B -- STABLE radix sort by coord_key. stable_sort_by_key keeps the
    // relative order of equal keys (the id order from PASS A), giving the total
    // order (coord_key asc, then id asc) == coord_less. Values = ids, so after
    // this d_ids is the permutation that puts reads in genome order.
    thrust::stable_sort_by_key(d_keys.begin(), d_keys.end(), d_ids.begin());

    *kernel_ms = timer.stop_ms();

    // GATHER on the host: d_ids now holds original ids in genome order. Copy it
    // down and pick the matching records. (A device gather is just as easy, but
    // this keeps the result assembly obvious; the timed work was the GPU sort.)
    thrust::host_vector<int> h_order = d_ids;
    for (int i = 0; i < n; ++i)
        out[static_cast<std::size_t>(i)] = rs.reads[static_cast<std::size_t>(h_order[i])];
}

// ===========================================================================
// (2) DUPLICATE MARKING
// ===========================================================================

// Map kernel: compute each read's duplicate signature key and carry its id.
//   Same one-thread-per-read mapping as coord_key_kernel. The (key,id) pairs are
//   then sorted so equal signatures are contiguous for the segmented reduction.
__global__ void dup_key_kernel(const ReadRecord* __restrict__ reads, int n,
                               uint64_t* __restrict__ keys,
                               int* __restrict__ ids) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    keys[i] = dup_key(reads[i]);   // bam.h: (ref,pos,strand,mate) -> one uint64
    ids[i]  = reads[i].id;
}

// Map kernel: write the final is_dup[] flags.
//   After the segmented reduction we know, for each read's group, the id of the
//   representative to KEEP (best quality). One thread per sorted slot sets
//   is_dup[id]=1 unless this read IS that representative. Each id appears in
//   exactly one slot, so writes hit distinct is_dup[] addresses -> no atomics.
__global__ void mark_kernel(int n,
                            const int* __restrict__ sorted_ids,        // ids in dup-sorted order
                            const int* __restrict__ group_of,          // group index per sorted slot
                            const int* __restrict__ keep_id_of_group,  // best id per group
                            uint8_t* __restrict__ is_dup) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;   // slot in sorted order
    if (s >= n) return;
    const int id   = sorted_ids[s];                 // this read's original id
    const int g    = group_of[s];                   // which signature-group it is in
    const int keep = keep_id_of_group[g];           // the kept representative's id
    is_dup[id] = (id == keep) ? 0u : 1u;            // duplicate unless it is the keeper
}

// Binary functor for reduce_by_key: given two reads (carried as their ids),
// return the id of the BETTER one (is_better_dup, bam.h). is_better_dup is a
// TOTAL order (score desc, then id asc), so this "argmax" is associative AND
// commutative -- the two properties reduce_by_key requires. That is precisely
// what makes the GPU's per-group winner order-independent and equal to the CPU's.
struct BestDup {
    const ReadRecord* reads;   // device pointer to the read array
    __host__ __device__
    int operator()(int id_a, int id_b) const {
        return is_better_dup(reads[id_a], reads[id_b]) ? id_a : id_b;
    }
};

int markdup_gpu(const ReadSet& rs, std::vector<uint8_t>& is_dup, float* kernel_ms) {
    const int n = rs.n();
    is_dup.assign(static_cast<std::size_t>(n), 0);

    thrust::device_vector<ReadRecord> d_reads(rs.reads.begin(), rs.reads.end());
    thrust::device_vector<uint64_t>   d_keys(n);    // dup_key per read
    thrust::device_vector<int>        d_ids(n);     // original id per read
    thrust::device_vector<uint8_t>    d_is_dup(n, 0);

    const ReadRecord* d_reads_ptr = thrust::raw_pointer_cast(d_reads.data());
    const int grid = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    GpuTimer timer;
    timer.start();

    // (a) MAP: compute the dup signature key and id for every read.
    dup_key_kernel<<<grid, THREADS_PER_BLOCK>>>(
        d_reads_ptr, n,
        thrust::raw_pointer_cast(d_keys.data()),
        thrust::raw_pointer_cast(d_ids.data()));
    CUDA_CHECK_LAST("dup_key_kernel");

    // (b) SORT by signature so equal-signature reads are contiguous. We carry
    //     the ids as the values. (The per-group winner is chosen by BestDup in
    //     step (c), which is order-independent, so the sort here only needs to
    //     GROUP equal keys, not order within a group.)
    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_ids.begin());
    // Now d_keys is non-decreasing; d_ids[s] is the read in sorted slot s.

    // (c) SEGMENTED REDUCTION: collapse each run of equal keys to ONE output --
    //     the id of that group's best read. thrust::reduce_by_key walks the
    //     sorted keys, and for each maximal run of EQUAL keys it reduces the
    //     corresponding values with our BestDup op, emitting (unique_key,
    //     best_id) once per group. This is the GPU's "GROUP BY signature, keep
    //     argmax(quality)" -- the core of MarkDuplicates.
    thrust::device_vector<uint64_t> d_group_keys(n);   // unique signatures (<= n)
    thrust::device_vector<int>      d_group_best(n);    // best id per group (<= n)
    BestDup best_op{ d_reads_ptr };
    auto ends = thrust::reduce_by_key(
        d_keys.begin(), d_keys.end(),   // keys (the sorted signatures)
        d_ids.begin(),                  // values: the ids in the same order
        d_group_keys.begin(),             // out: one unique key per group
        d_group_best.begin(),             // out: best id per group
        cuda::std::equal_to<uint64_t>(),  // groups = maximal runs of equal key
        best_op);                         // combine: keep the better-dup id
    const int num_groups = static_cast<int>(ends.first - d_group_keys.begin());

    // (d) For each sorted slot, which GROUP does it belong to? Because the keys
    //     are sorted, a new group starts wherever the key changes. We build a
    //     "new-group flag" (1 at each boundary, 0 elsewhere) and an inclusive
    //     scan turns it into the group index for every slot (0,0,1,1,1,2,...) --
    //     a standard segmented-id trick. reduce_by_key emits groups in this same
    //     left-to-right order, so group index == row in d_group_best.
    thrust::device_vector<int> d_group_of(n);
    thrust::device_vector<int> d_flags(n);
    // flags[s] = (keys[s] != keys[s-1]) for s>=1, computed by transform on the
    // two shifted views of d_keys; flags[0] is set to 0 separately.
    thrust::transform(
        d_keys.begin() + 1, d_keys.end(),       // keys[1..n-1]
        d_keys.begin(),                         // keys[0..n-2]
        d_flags.begin() + 1,                    // write to flags[1..n-1]
        cuda::std::not_equal_to<uint64_t>());   // 1 where the key changed
    d_flags[0] = 0;                          // first slot is always group 0
    thrust::inclusive_scan(d_flags.begin(), d_flags.end(), d_group_of.begin());

    // (e) MARK: one thread per sorted slot writes is_dup[id]. Each id appears in
    //     exactly one slot, so the writes are to distinct addresses (no atomics).
    mark_kernel<<<grid, THREADS_PER_BLOCK>>>(
        n,
        thrust::raw_pointer_cast(d_ids.data()),
        thrust::raw_pointer_cast(d_group_of.data()),
        thrust::raw_pointer_cast(d_group_best.data()),
        thrust::raw_pointer_cast(d_is_dup.data()));
    CUDA_CHECK_LAST("mark_kernel");

    *kernel_ms = timer.stop_ms();

    // Copy flags down and count duplicates (the deterministic headline number).
    thrust::host_vector<uint8_t> h_is_dup = d_is_dup;
    int dups = 0;
    for (int i = 0; i < n; ++i) {
        is_dup[static_cast<std::size_t>(i)] = h_is_dup[i];
        dups += h_is_dup[i];
    }
    (void)num_groups;   // groups == n - dups; kept for debugging clarity
    return dups;
}
