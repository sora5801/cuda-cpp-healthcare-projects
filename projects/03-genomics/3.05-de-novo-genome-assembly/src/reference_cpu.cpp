// ===========================================================================
// src/reference_cpu.cpp  --  FASTA loader, minimizer sketcher, serial overlap
// ---------------------------------------------------------------------------
// Project 3.5 : De Novo Genome Assembly  (all-vs-all read-overlap stage)
//
// ROLE IN THE PROJECT
//   Three plain-C++ jobs, all compiled by the host compiler (no CUDA here):
//     (1) load_fasta()  : parse the tiny FASTA-like sample into read strings.
//     (2) sketch_reads(): turn each read into its SORTED, UNIQUE minimizer set
//                         (the cheap, inherently-serial pre-processing step).
//     (3) overlap_cpu() : the OBVIOUSLY-correct serial all-vs-all comparison the
//                         GPU kernel is verified against.
//   The per-pair scoring math itself lives in assembly.h (count_shared_sorted),
//   shared verbatim with the GPU -> exact agreement (PATTERNS.md sec.2).
//
// READ THIS AFTER: assembly.h, reference_cpu.h. Compare overlap_cpu() with the
// GPU twin overlap_kernel() in kernels.cu.
// ===========================================================================
#include "reference_cpu.h"

#include <algorithm>   // std::sort, std::unique, std::min/max
#include <cctype>      // std::toupper
#include <fstream>     // std::ifstream
#include <stdexcept>   // std::runtime_error

// ---------------------------------------------------------------------------
// base_code: map a DNA letter to its 2-bit code (A=0,C=1,G=2,T=3), or -1 for
// anything else (N, gaps, whitespace). Packing 2 bits/base lets a K=15 k-mer
// fit in a 30-bit integer, so a whole k-mer is one cheap machine word.
// ---------------------------------------------------------------------------
static inline int base_code(char c) {
    switch (std::toupper(static_cast<unsigned char>(c))) {
        case 'A': return 0;
        case 'C': return 1;
        case 'G': return 2;
        case 'T': return 3;
        default:  return -1;   // ambiguous/invalid base -> breaks the k-mer run
    }
}

// complement of a 2-bit base code: A<->T (0<->3), C<->G (1<->2). Used to build
// the reverse-complement k-mer so we sketch the CANONICAL (strand-independent)
// k-mer -- essential because a read and its overlapping neighbour may come from
// opposite DNA strands (THEORY "The science: double-stranded DNA").
static inline std::uint32_t comp_code(std::uint32_t b) { return 3u - b; }

// ---------------------------------------------------------------------------
// load_fasta: read a FASTA-like file into a vector of sequence strings.
//   We accept the minimal subset our samples use: '>' header lines (whose text
//   we ignore beyond counting a new record) and sequence lines. Multiple
//   sequence lines for one record are concatenated. Empty result -> throw.
// ---------------------------------------------------------------------------
std::vector<std::string> load_fasta(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open reads file: " + path);

    std::vector<std::string> reads;
    std::string line, cur;
    auto flush = [&]() { if (!cur.empty()) { reads.push_back(cur); cur.clear(); } };
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();   // CRLF safety
        if (line.empty()) continue;
        if (line[0] == '>') { flush(); continue; }   // header -> start new record
        cur += line;                                  // accumulate sequence bases
    }
    flush();   // last record
    if (reads.empty())
        throw std::runtime_error("no reads parsed from " + path + " (empty/invalid FASTA)");
    return reads;
}

// ---------------------------------------------------------------------------
// canonical_kmers: roll a K-base window across one read, emitting the canonical
// 2-bit-packed k-mer at each position. The canonical k-mer is min(forward,
// reverse-complement) so the same physical DNA gives the same key regardless of
// which strand the read came from.
//   We maintain TWO rolling registers updated in O(1) per base:
//     fwd = (fwd << 2 | code) & mask          -- shift in the new base
//     rev = (rev >> 2) | (comp(code) << top)  -- prepend the complement
//   A run of valid bases must reach length K before the first k-mer is emitted;
//   any invalid base (N) resets the run. This is the textbook minimizer-sketch
//   inner loop (minimap2's `mm_sketch`), kept readable.
//   out : receives one packed canonical k-mer per valid window position.
// ---------------------------------------------------------------------------
static void canonical_kmers(const std::string& seq, std::vector<std::uint32_t>& out) {
    out.clear();
    const std::uint32_t mask = (K < 16) ? ((1u << (2 * K)) - 1u) : 0xffffffffu;  // low 2K bits
    const int top_shift = 2 * (K - 1);   // where the reverse register's new base lands
    std::uint32_t fwd = 0, rev = 0;
    int run = 0;                          // consecutive valid bases so far
    for (char ch : seq) {
        const int c = base_code(ch);
        if (c < 0) { run = 0; fwd = 0; rev = 0; continue; }   // reset on ambiguous base
        fwd = ((fwd << 2) | static_cast<std::uint32_t>(c)) & mask;
        rev = (rev >> 2) | (comp_code(static_cast<std::uint32_t>(c)) << top_shift);
        if (++run >= K) out.push_back(std::min(fwd, rev));    // canonical = smaller register
    }
}

// ---------------------------------------------------------------------------
// minimizers_of: turn a read's k-mer stream into its minimizer SKETCH.
//   For each window of W consecutive k-mers we keep the one with the smallest
//   HASH (hash32 in assembly.h). Adjacent windows overlap, so the same k-mer is
//   often picked repeatedly; we collect all picks, then SORT + UNIQUE so the
//   final per-read list is sorted and deduplicated -- exactly the precondition
//   count_shared_sorted() needs. (A real minimizer index keeps positions too;
//   for overlap *detection* the set of distinct minimizers suffices.)
// ---------------------------------------------------------------------------
static void minimizers_of(const std::string& seq, std::vector<minimizer_t>& sketch) {
    sketch.clear();
    std::vector<std::uint32_t> kmers;
    canonical_kmers(seq, kmers);
    const int m = static_cast<int>(kmers.size());
    if (m == 0) return;

    if (m < W) {
        // Read too short for a full window: take the single global minimum so
        // even short reads contribute one minimizer (keeps the loop total simple).
        std::uint32_t best = hash32(kmers[0]);
        std::uint32_t bestk = kmers[0];
        for (int t = 1; t < m; ++t) {
            std::uint32_t h = hash32(kmers[t]);
            if (h < best) { best = h; bestk = kmers[t]; }
        }
        sketch.push_back(bestk);
    } else {
        // Slide a length-W window; emit the min-hash k-mer of each window. O(m*W)
        // here for clarity; THEORY notes the O(m) monotonic-deque version used at
        // scale (the deque is left as an exercise -- the result is identical).
        for (int s = 0; s + W <= m; ++s) {
            std::uint32_t best = hash32(kmers[s]);
            std::uint32_t bestk = kmers[s];
            for (int t = 1; t < W; ++t) {
                std::uint32_t h = hash32(kmers[s + t]);
                if (h < best) { best = h; bestk = kmers[s + t]; }
            }
            sketch.push_back(bestk);
        }
    }
    // Sort + unique so the sketch is a sorted SET (count_shared_sorted's input).
    std::sort(sketch.begin(), sketch.end());
    sketch.erase(std::unique(sketch.begin(), sketch.end()), sketch.end());
}

// ---------------------------------------------------------------------------
// sketch_reads: build the flattened CSR ReadSet the GPU consumes (declared in
// assembly.h). We sketch every read, then concatenate the sorted-unique sketches
// into one buffer `mins` with an `offset` prefix-sum so any read's slice is
// mins[offset[r] .. offset[r+1]). This ragged-array layout is GPU-friendly: a
// thread finds its read's minimizers in O(1) with no jagged 2-D structure.
// ---------------------------------------------------------------------------
ReadSet sketch_reads(const std::vector<std::string>& reads) {
    ReadSet rs;
    rs.n = static_cast<int>(reads.size());
    rs.offset.assign(static_cast<std::size_t>(rs.n) + 1, 0);
    rs.read_len.assign(static_cast<std::size_t>(rs.n), 0);

    std::vector<minimizer_t> sk;
    for (int r = 0; r < rs.n; ++r) {
        rs.read_len[r] = static_cast<int>(reads[r].size());
        minimizers_of(reads[r], sk);
        // Append this read's sketch and record where the NEXT read starts.
        rs.mins.insert(rs.mins.end(), sk.begin(), sk.end());
        rs.offset[r + 1] = static_cast<int>(rs.mins.size());
    }
    return rs;
}

// ---------------------------------------------------------------------------
// overlap_cpu: the serial all-vs-all reference. Loop every unordered pair (i<j),
// count shared minimizers with the SHARED routine, and record edges that clear
// MIN_SHARED. We enumerate pairs in the SAME flat order pair_to_ij() decodes, so
// out_score_all lines up index-for-index with the GPU's per-pair scores.
//   Complexity: O(n^2) pairs * O(sketch length) each. This is the bottleneck the
//   GPU parallelizes (one thread per pair) -- see kernels.cu.
// ---------------------------------------------------------------------------
void overlap_cpu(const ReadSet& rs,
                 std::vector<Overlap>& overlaps,
                 std::vector<int>* out_score_all) {
    overlaps.clear();
    const long long P = num_pairs(rs.n);
    if (out_score_all) out_score_all->assign(static_cast<std::size_t>(P), 0);

    for (long long p = 0; p < P; ++p) {
        int i, j;
        pair_to_ij(p, rs.n, &i, &j);                 // decode flat index -> (i<j)
        const minimizer_t* a = rs.mins.data() + rs.offset[i];
        const minimizer_t* b = rs.mins.data() + rs.offset[j];
        const int na = rs.offset[i + 1] - rs.offset[i];
        const int nb = rs.offset[j + 1] - rs.offset[j];
        const int shared = count_shared_sorted(a, na, b, nb);   // THE shared math
        if (out_score_all) (*out_score_all)[static_cast<std::size_t>(p)] = shared;
        if (shared >= MIN_SHARED) overlaps.push_back(Overlap{i, j, shared});
    }
    // overlaps are already in (i,j) order because p enumerates the triangle
    // row by row; no extra sort needed -> deterministic output.
}
