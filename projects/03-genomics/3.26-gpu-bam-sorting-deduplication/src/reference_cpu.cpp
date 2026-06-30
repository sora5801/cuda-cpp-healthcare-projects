// ===========================================================================
// src/reference_cpu.cpp  --  The plain-C++ baseline we trust
// ---------------------------------------------------------------------------
// Project 3.26 : GPU BAM Sorting & Deduplication
//
// ROLE IN THE PROJECT
//   This is the "ground truth" the GPU result is checked against. It is written
//   to be OBVIOUSLY correct -- ordinary std::sort and a single grouping loop, no
//   parallelism, no cleverness -- so that when the GPU and CPU agree, we believe
//   the GPU. Compiled by the host C++ compiler only (no CUDA here).
//
//   Both operations call the SAME comparison helpers (coord_less, dup_key,
//   is_better_dup) the GPU uses, all defined in bam.h. Because every comparison
//   is on integers and every order is made total by the `id` tie-break, the CPU
//   and GPU produce byte-identical results -- verification is EXACT (no epsilon).
//
// READ THIS AFTER: bam.h, reference_cpu.h. Compare against kernels.cu (GPU twin).
// ===========================================================================
#include "reference_cpu.h"

#include <algorithm>    // std::sort
#include <fstream>      // std::ifstream
#include <stdexcept>    // std::runtime_error
#include <string>
#include <unordered_map>

// ---------------------------------------------------------------------------
// load_readset -- parse the tiny text read set (format in data/README.md).
//   Validates ranges so the 24-bit/15-bit key packing in bam.h cannot overflow
//   and silently corrupt the sort order. Assigns each record's `id` from its
//   line order, which is the total-order tie-breaker used everywhere.
// ---------------------------------------------------------------------------
ReadSet load_readset(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open read set file: " + path);

    ReadSet rs;
    int n = 0;
    if (!(in >> n >> rs.num_refs) || n <= 0 || rs.num_refs <= 0)
        throw std::runtime_error("bad header (expected '<n> <num_refs>') in " + path);

    rs.reads.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        ReadRecord r{};
        if (!(in >> r.ref_id >> r.pos >> r.strand >> r.mate_pos >> r.base_qual_sum))
            throw std::runtime_error("read set truncated at record " + std::to_string(i)
                                     + " in " + path);
        // Range checks matching the bit budgets in bam.h (24/24/1/15 bits).
        if (r.ref_id < 0 || r.ref_id >= rs.num_refs)
            throw std::runtime_error("ref_id out of range at record " + std::to_string(i));
        if (r.pos < 0 || r.pos > 0xFFFFFF)
            throw std::runtime_error("pos out of 24-bit range at record " + std::to_string(i));
        if (r.strand != 0 && r.strand != 1)
            throw std::runtime_error("strand must be 0 or 1 at record " + std::to_string(i));
        if (r.mate_pos < 0 || r.mate_pos > 0x7FFF)
            throw std::runtime_error("mate_pos out of 15-bit range at record " + std::to_string(i));
        if (r.base_qual_sum < 0)
            throw std::runtime_error("base_qual_sum must be >= 0 at record " + std::to_string(i));
        r.id = i;                       // original input index = total-order tiebreak
        rs.reads.push_back(r);
    }
    return rs;
}

// ---------------------------------------------------------------------------
// sort_cpu -- coordinate-sort with the shared total order coord_less.
//   We copy the reads and std::sort them. std::sort is not guaranteed stable,
//   but coord_less is a TOTAL order (it tie-breaks on the unique `id`), so the
//   output is uniquely determined regardless of the sort's internal pivoting --
//   the same property that lets the GPU radix sort match us exactly.
// ---------------------------------------------------------------------------
void sort_cpu(const ReadSet& rs, std::vector<ReadRecord>& out) {
    out = rs.reads;                                  // copy input-order reads
    std::sort(out.begin(), out.end(), coord_less);   // genome order (bam.h)
}

// ---------------------------------------------------------------------------
// markdup_cpu -- flag PCR/optical duplicates by grouping on the dup signature.
//
//   ALGORITHM (the readable serial version of what the GPU does in parallel):
//     1. For each duplicate signature (dup_key), remember the BEST read seen so
//        far (highest base-quality sum; ties -> lowest id, via is_better_dup).
//        A hash map signature -> best-so-far record does this in O(n).
//     2. After one pass, every signature's map entry is its single representative
//        (the "original"). Any read that is NOT its group's representative is a
//        duplicate.
//     3. Walk the reads again; a read is a duplicate iff it differs (by id) from
//        its group's representative. Write is_dup[read.id].
//
//   Determinism: is_better_dup is a total order on (score desc, id asc), so the
//   representative of each group is unique. The GPU reaches the SAME choice via
//   a segmented reduction, so is_dup matches exactly. O(n) expected time.
// ---------------------------------------------------------------------------
int markdup_cpu(const ReadSet& rs, std::vector<uint8_t>& is_dup) {
    const int n = rs.n();
    is_dup.assign(static_cast<std::size_t>(n), 0);

    // Pass 1: signature -> id of the best (kept) read in that group.
    std::unordered_map<uint64_t, int> best_id;
    best_id.reserve(static_cast<std::size_t>(n) * 2);
    for (const ReadRecord& r : rs.reads) {
        const uint64_t k = dup_key(r);
        auto it = best_id.find(k);
        if (it == best_id.end()) {
            best_id.emplace(k, r.id);                // first read of this group
        } else if (is_better_dup(r, rs.reads[static_cast<std::size_t>(it->second)])) {
            it->second = r.id;                       // this read is a better keep
        }
    }

    // Pass 2: a read is a duplicate iff it is not its group's representative.
    int dups = 0;
    for (const ReadRecord& r : rs.reads) {
        const int keep = best_id[dup_key(r)];
        if (r.id != keep) { is_dup[static_cast<std::size_t>(r.id)] = 1; ++dups; }
    }
    return dups;
}
