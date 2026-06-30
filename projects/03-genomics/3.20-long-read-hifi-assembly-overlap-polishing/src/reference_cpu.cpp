// ===========================================================================
// src/reference_cpu.cpp  --  The trusted serial overlap baseline + sketch loader
// ---------------------------------------------------------------------------
// Project 3.20 : Long-Read HiFi Assembly Overlap & Polishing
//
// ROLE IN THE PROJECT
//   (1) load_reads()          : parse the minimiser-sketch dataset (data/README).
//   (2) chain_overlap_score() : for ONE read pair, build the shared-seed anchors
//                               and run the collinear chaining DP -> overlap score.
//   (3) overlap_cpu()         : loop that step over every (i<j) pair, in order.
//
//   This is the "ground truth" the GPU result is checked against. It is written
//   to be OBVIOUSLY correct -- single readable loops, no parallelism -- so that
//   when the GPU and CPU agree we believe the GPU. The per-link scoring it uses
//   (ovl_chain_link_score) is the SAME function the GPU kernel calls, from
//   overlap_core.h, and every score is an INTEGER, so the two sides match
//   bit-for-bit (../THEORY.md "How we verify correctness").
//
//   Compiled by the host C++ compiler only (no CUDA here). See reference_cpu.h.
//
// READ THIS AFTER: reference_cpu.h, overlap_core.h. Compare against kernels.cu.
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
// load_reads: read the sketched dataset. Format (data/README.md):
//   line 1                : "<n_reads>"
//   for each read r       : "<read_len> <cnt>"
//   then cnt lines        : "<pos> <hash-as-8-hex-digits>"   (pos ascending)
// We build the flat ReadSet (concatenated minimisers + per-read offset/count).
// ---------------------------------------------------------------------------
ReadSet load_reads(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open read-sketch file: " + path);

    int n_reads = 0;
    if (!(in >> n_reads) || n_reads <= 0)
        throw std::runtime_error("bad header (expected positive <n_reads>) in " + path);

    ReadSet rs;
    rs.n_reads = n_reads;
    rs.read_len.resize(n_reads);
    rs.off.resize(n_reads);
    rs.cnt.resize(n_reads);

    // Read each record, appending its minimisers to the flat `mins` array and
    // recording where this read's slice starts (off) and how long it is (cnt).
    for (int r = 0; r < n_reads; ++r) {
        int rlen = 0, cnt = 0;
        if (!(in >> rlen >> cnt) || rlen <= 0 || cnt < 0)
            throw std::runtime_error("malformed read header at read " + std::to_string(r)
                                     + " in " + path);
        rs.read_len[r] = rlen;
        rs.off[r]      = static_cast<int32_t>(rs.mins.size());
        rs.cnt[r]      = cnt;

        int32_t prev_pos = -1;   // to assert the per-read pos-ascending invariant
        for (int m = 0; m < cnt; ++m) {
            long pos = 0;
            std::string hex;
            if (!(in >> pos >> hex))
                throw std::runtime_error("unexpected end of minimisers at read "
                                         + std::to_string(r) + " in " + path);
            // hashes are written as hex (no 0x); std::stoul base 16 parses them.
            const uint32_t hash = static_cast<uint32_t>(std::stoul(hex, nullptr, 16));
            if (static_cast<int32_t>(pos) < prev_pos)
                throw std::runtime_error("minimisers not pos-sorted at read "
                                         + std::to_string(r) + " in " + path);
            prev_pos = static_cast<int32_t>(pos);
            Minimizer mz;
            mz.pos  = static_cast<int32_t>(pos);
            mz.hash = hash;
            rs.mins.push_back(mz);
        }
    }
    return rs;
}

// ---------------------------------------------------------------------------
// chain_overlap_score: score ONE read pair.
//
// STEP A -- build shared-seed anchors.
//   An anchor is a (query_pos, target_pos) pair where the query and target reads
//   carry the SAME minimiser hash. We walk the query minimisers in pos order
//   (outer loop) and, for each, scan the target minimisers in pos order (inner
//   loop) for a hash match. This emits anchors QUERY-POS-MAJOR (and, within one
//   query hash, target-pos-ascending). That ordering is exactly what the chaining
//   DP below needs (it only links a later anchor to an earlier one in this order).
//   We cap the anchor list at OVL_MAX_ANCHORS so the work per pair is bounded --
//   the GPU twin uses the identical cap so both sides keep the same anchors.
//
// STEP B -- collinear chaining DP.
//   Classic 1-D chaining (the minimap2 / Li 2018 recurrence, simplified):
//       f[a] = MATCH_AWARD + max( 0, max over b<a of ( f[b] + link(b -> a) ) )
//   f[a] is the best chain score ENDING at anchor a. A link b->a is allowed only
//   if both coordinates strictly increase and the diagonal drift is small
//   (ovl_chain_link_score returns OVL_REJECT otherwise). The answer is max_a f[a].
//   Complexity O(A^2) in the anchor count A (A <= OVL_MAX_ANCHORS, so bounded).
//
//   Everything is integer, so this is bit-identical to the device version.
// ---------------------------------------------------------------------------
int chain_overlap_score(const Minimizer* q_min, int q_cnt,
                        const Minimizer* t_min, int t_cnt,
                        int* out_n_anchors) {
    // --- STEP A: collect anchors (query-pos-major) ---
    int  anchor_q[OVL_MAX_ANCHORS];   // query positions of the anchors
    int  anchor_t[OVL_MAX_ANCHORS];   // target positions of the anchors
    int  n_anchor = 0;
    for (int qi = 0; qi < q_cnt && n_anchor < OVL_MAX_ANCHORS; ++qi) {
        const uint32_t h = q_min[qi].hash;
        for (int ti = 0; ti < t_cnt && n_anchor < OVL_MAX_ANCHORS; ++ti) {
            if (t_min[ti].hash == h) {
                anchor_q[n_anchor] = q_min[qi].pos;
                anchor_t[n_anchor] = t_min[ti].pos;
                ++n_anchor;
            }
        }
    }
    if (out_n_anchors) *out_n_anchors = n_anchor;
    if (n_anchor == 0) return 0;   // no shared seeds -> no overlap evidence

    // --- STEP B: collinear chaining, BOTH strands (overlap_core.h) ---
    // ovl_chain_best_both_strands runs the O(A^2) DP twice -- once for a forward
    // overlap, once for a reverse-complement overlap (target axis negated) -- and
    // returns the stronger chain. This is the SAME function the GPU kernel calls,
    // so the integer scores are bit-identical. `f` and `neg` are scratch buffers.
    int f[OVL_MAX_ANCHORS];        // DP table: f[a] = best chain score ending at a
    int neg[OVL_MAX_ANCHORS];      // scratch for the strand-flipped target coords
    return ovl_chain_best_both_strands(anchor_q, anchor_t, n_anchor, f, neg);
}

// ---------------------------------------------------------------------------
// overlap_cpu: enumerate all ordered pairs (i<j) and score each one, writing the
//   results into `out` in pair_index order (the SAME order the GPU kernel uses).
//   This is the serial baseline whose runtime makes the GPU speed-up legible.
// ---------------------------------------------------------------------------
void overlap_cpu(const ReadSet& rs, std::vector<OverlapResult>& out) {
    const long long P = rs.num_pairs();
    out.assign(static_cast<std::size_t>(P), OverlapResult{});
    const int N = rs.n_reads;

    for (int i = 0; i < N; ++i) {
        const Minimizer* qi = rs.mins.data() + rs.off[i];   // read i's slice
        const int qc = rs.cnt[i];
        for (int j = i + 1; j < N; ++j) {
            const Minimizer* tj = rs.mins.data() + rs.off[j];   // read j's slice
            const int tc = rs.cnt[j];
            int n_anchor = 0;
            const int score = chain_overlap_score(qi, qc, tj, tc, &n_anchor);

            OverlapResult res;
            res.read_i    = i;
            res.read_j    = j;
            res.score     = score;
            res.n_anchors = n_anchor;
            out[static_cast<std::size_t>(pair_index(i, j, N))] = res;
        }
    }
}
