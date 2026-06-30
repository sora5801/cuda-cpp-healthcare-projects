// ===========================================================================
// src/reference_cpu.cpp  --  Trusted serial MSA: loader, distances, assembly
// ---------------------------------------------------------------------------
// Project 3.8 : Multiple Sequence Alignment (MSA)
//
// The plain-C++ baseline the GPU is checked against. It is written to be
// OBVIOUSLY correct -- readable loops, no parallelism -- so that when the GPU
// pairwise-score matrix agrees with this one (exact integer match), we trust it.
//
//   (1) load_fasta()          : parse the tiny multi-FASTA sample.
//   (2) distance_matrix_cpu() : STAGE 1 -- NW score every pair (the GPU twin of
//                               this is kernels.cu); derive the distance matrix.
//   (3) build_msa()           : STAGES 2-3 -- center-star pick + progressive
//                               assembly into the final alignment.
//   (4) sum_of_pairs()        : grade the assembled alignment.
//
// Compiled by the host C++ compiler only (no CUDA here). See reference_cpu.h and
// the shared recurrence in nw_core.h.
// ===========================================================================
#include "reference_cpu.h"

#include <algorithm>
#include <fstream>
#include <stdexcept>

// Public alphabet table (declared extern in reference_cpu.h).
const char DNA_ALPHABET[5] = {'A', 'C', 'G', 'T', '\0'};

// ---------------------------------------------------------------------------
// encode: map a nucleotide character to its 0..3 code, or throw on a bad letter.
//   We accept upper/lower case. Anything else (N, gaps, protein letters, ...) is
//   rejected loudly: this teaching project handles clean DNA only.
// ---------------------------------------------------------------------------
static uint8_t encode(char c) {
    switch (c) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'T': case 't': return 3;
        default:
            throw std::runtime_error(std::string("non-ACGT character in sequence: '") + c + "'");
    }
}

// ---------------------------------------------------------------------------
// load_fasta: read a multi-FASTA file into the flat SeqSet layout.
//   FASTA = a ">header" line, then one or more sequence lines, repeated. We
//   accumulate residues until the next '>' (or EOF), then commit one sequence.
//   The flat `data`/`off`/`len` layout (see reference_cpu.h) is built as we go.
// ---------------------------------------------------------------------------
SeqSet load_fasta(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open FASTA file: " + path);

    SeqSet s;
    std::string line;
    std::string cur_name;            // header of the sequence we are reading
    std::vector<uint8_t> cur_res;    // its residues so far (encoded)
    bool have = false;               // are we inside a record?

    // commit(): finalise the current record into the flat buffers.
    auto commit = [&]() {
        if (!have) return;
        if (cur_res.empty())
            throw std::runtime_error("empty sequence for record '" + cur_name + "' in " + path);
        s.off.push_back(static_cast<int>(s.data.size()));   // start = current end
        s.len.push_back(static_cast<int>(cur_res.size()));
        s.names.push_back(cur_name);
        s.data.insert(s.data.end(), cur_res.begin(), cur_res.end());
        s.max_len = std::max(s.max_len, static_cast<int>(cur_res.size()));
        cur_res.clear();
    };

    while (std::getline(in, line)) {
        // Strip a trailing CR so Windows-edited files parse on any platform.
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;                 // skip blank lines
        if (line[0] == '>') {                       // header -> start a new record
            commit();                               // finish the previous one first
            cur_name = line.substr(1);              // drop the '>'
            have = true;
            continue;
        }
        if (!have) throw std::runtime_error("sequence data before any '>' header in " + path);
        for (char c : line) {
            if (c == ' ' || c == '\t') continue;    // tolerate stray whitespace
            cur_res.push_back(encode(c));
        }
    }
    commit();                                       // commit the final record

    s.n = static_cast<int>(s.len.size());
    if (s.n < 2) throw std::runtime_error("need at least 2 sequences for an MSA in " + path);
    return s;
}

// ---------------------------------------------------------------------------
// distance_matrix_cpu: STAGE 1. NW-score every pair (a,b) with a<=b, mirror the
// result into the symmetric matrix, and derive the normalised distance.
//
//   Per pair we call nw_score_core() (the SHARED recurrence in nw_core.h, the
//   very same function the GPU kernel calls). Two rolling DP rows of size
//   (max_len+1) are reused across all pairs -- one allocation, not N^2.
//
//   Distance normalisation: the maximum possible score for sequence a is its
//   self-score (all matches, no gaps) = la*MATCH. We normalise by the SMALLER
//   of the two self-scores so identical sequences give distance 0 and very
//   dissimilar ones approach 1. Distance is clamped to [0,1]. (THEORY explains
//   why this simple normalisation suffices for a guide order.)
// ---------------------------------------------------------------------------
void distance_matrix_cpu(const SeqSet& s,
                         std::vector<int>& raw_score,
                         std::vector<double>& D) {
    const int n = s.n;
    raw_score.assign(static_cast<std::size_t>(n) * n, 0);
    D.assign(static_cast<std::size_t>(n) * n, 0.0);

    std::vector<int> prev(s.max_len + 1), curr(s.max_len + 1);   // rolling DP rows

    for (int a = 0; a < n; ++a) {
        for (int b = a; b < n; ++b) {
            const int sc = nw_score_core(s.seq(a), s.len[a],
                                         s.seq(b), s.len[b],
                                         prev.data(), curr.data());
            raw_score[static_cast<std::size_t>(a) * n + b] = sc;
            raw_score[static_cast<std::size_t>(b) * n + a] = sc;   // symmetric

            // Normalised distance. self = best achievable score for the shorter
            // sequence; guard self<=0 (degenerate) by treating it as distance 0.
            const int self = nw_self_score(std::min(s.len[a], s.len[b]));
            double dist = (self > 0) ? (1.0 - static_cast<double>(sc) / self) : 0.0;
            if (dist < 0.0) dist = 0.0;
            if (dist > 1.0) dist = 1.0;
            D[static_cast<std::size_t>(a) * n + b] = dist;
            D[static_cast<std::size_t>(b) * n + a] = dist;
        }
    }
}

// ---------------------------------------------------------------------------
// nw_align_pair: full Needleman-Wunsch alignment of a vs b WITH traceback.
//   Unlike nw_score_core (score only, O(L) memory), here we need the actual gap
//   placement, so we materialise the (la+1)*(lb+1) matrix and walk back from the
//   corner. Returns the two aligned strings (as 0..4 codes, 4 = gap) of equal
//   length. Used only by build_msa() on the host -- assembly is bookkeeping, not
//   the GPU lesson, so the simple O(L^2)-memory version is fine here.
//   Deterministic tie-break: diagonal > up > left (matches traceback in 3.01).
// ---------------------------------------------------------------------------
namespace {
constexpr uint8_t GAP_CODE = 4;     // sentinel residue meaning "gap" in a row

struct PairAln { std::vector<uint8_t> a, b; };   // aligned, equal length

PairAln nw_align_pair(const uint8_t* a, int la, const uint8_t* b, int lb) {
    const int W = lb + 1;
    std::vector<int> H(static_cast<std::size_t>(la + 1) * W);
    // Boundary: leading gaps cost i*GAP / j*GAP (same seeding as nw_core.h).
    for (int i = 0; i <= la; ++i) H[static_cast<std::size_t>(i) * W + 0] = i * NW_GAP;
    for (int j = 0; j <= lb; ++j) H[j] = j * NW_GAP;
    for (int i = 1; i <= la; ++i) {
        for (int j = 1; j <= lb; ++j) {
            const int diag = H[static_cast<std::size_t>(i - 1) * W + (j - 1)] + nw_subst(a[i - 1], b[j - 1]);
            const int up   = H[static_cast<std::size_t>(i - 1) * W + j]       + NW_GAP;
            const int left = H[static_cast<std::size_t>(i) * W + (j - 1)]     + NW_GAP;
            int v = diag; if (up > v) v = up; if (left > v) v = left;
            H[static_cast<std::size_t>(i) * W + j] = v;
        }
    }
    // Traceback from the corner (la,lb) to (0,0), recording each column.
    PairAln out;
    int i = la, j = lb;
    while (i > 0 || j > 0) {
        if (i > 0 && j > 0 &&
            H[static_cast<std::size_t>(i) * W + j] ==
                H[static_cast<std::size_t>(i - 1) * W + (j - 1)] + nw_subst(a[i - 1], b[j - 1])) {
            out.a.push_back(a[i - 1]); out.b.push_back(b[j - 1]); --i; --j;        // diagonal
        } else if (i > 0 &&
            H[static_cast<std::size_t>(i) * W + j] ==
                H[static_cast<std::size_t>(i - 1) * W + j] + NW_GAP) {
            out.a.push_back(a[i - 1]); out.b.push_back(GAP_CODE); --i;             // up: gap in b
        } else {
            out.a.push_back(GAP_CODE); out.b.push_back(b[j - 1]); --j;             // left: gap in a
        }
    }
    std::reverse(out.a.begin(), out.a.end());     // we built it end -> start
    std::reverse(out.b.begin(), out.b.end());
    return out;
}
}  // namespace

// ---------------------------------------------------------------------------
// build_msa: STAGES 2-3. Choose the center-star sequence, then fold every other
// sequence onto it.
//
//   STAGE 2 (center): the center c minimises the total distance to all others,
//   sum_b D[c][b]. Deterministic tie-break: smallest index. The center becomes
//   the fixed "spine" every other sequence is aligned against.
//
//   STAGE 3 (progressive merge): align each non-center sequence to the center
//   with NW. Different pairwise alignments insert gaps into the center at
//   different places; we merge them with the classic "once a gap, always a gap"
//   rule -- the union of all gap positions in the center defines the final column
//   layout, and each row is threaded through that layout. We implement this by
//   building, for the center, the SUPERSET of gaps, then re-threading every
//   pairwise alignment into it. This is the heart of progressive MSA, done
//   simply and deterministically. THEORY.md walks a worked example.
// ---------------------------------------------------------------------------
MSA build_msa(const SeqSet& s, const std::vector<double>& D) {
    const int n = s.n;

    // ---- STAGE 2: pick the center (min total distance, lowest index wins) ----
    int center = 0;
    double best = 1e300;
    for (int c = 0; c < n; ++c) {
        double tot = 0.0;
        for (int b = 0; b < n; ++b) tot += D[static_cast<std::size_t>(c) * n + b];
        if (tot < best - 1e-12) { best = tot; center = c; }
    }

    // ---- STAGE 3a: pairwise-align every sequence to the center ---------------
    // For each non-center b we get (center', b') aligned. center' is the center
    // with some gaps; we must reconcile all the different center' gapping.
    std::vector<PairAln> alns(n);
    for (int b = 0; b < n; ++b) {
        if (b == center) continue;
        alns[b] = nw_align_pair(s.seq(center), s.len[center], s.seq(b), s.len[b]);
    }

    // ---- STAGE 3b: build the merged center coordinate frame ------------------
    // We represent the final center spine as a list of "profile columns". Walk
    // the original center residues 0..Lc-1; before consuming residue k, we may
    // need to insert however many gap-columns the various pairwise alignments
    // demanded just BEFORE center residue k. We track, per pairwise alignment,
    // how many gaps it inserts into the center before each residue.
    const int Lc = s.len[center];

    // gaps_before[b][k] = number of gap columns alignment b inserts into the
    // center immediately before center residue k (k in 0..Lc; k==Lc = trailing).
    std::vector<std::vector<int>> gaps_before(n, std::vector<int>(Lc + 1, 0));
    for (int b = 0; b < n; ++b) {
        if (b == center) continue;
        int k = 0;                                  // next center residue index
        int run = 0;                                // consecutive center-gaps seen
        const auto& ca = alns[b].a;                 // the center row of this pair
        for (uint8_t code : ca) {
            if (code == GAP_CODE) { ++run; }        // a gap inserted INTO center
            else { gaps_before[b][k] += run; run = 0; ++k; }
        }
        gaps_before[b][Lc] += run;                  // trailing gaps after last residue
    }

    // The MERGED number of gap columns before center residue k is the MAX over
    // all pairwise alignments (every alignment's gaps must fit -> take the union,
    // which for simple insertions is the max). This yields the final width.
    std::vector<int> merged_gaps(Lc + 1, 0);
    for (int k = 0; k <= Lc; ++k) {
        int mx = 0;
        for (int b = 0; b < n; ++b) if (b != center) mx = std::max(mx, gaps_before[b][k]);
        merged_gaps[k] = mx;
    }

    int width = Lc;
    for (int k = 0; k <= Lc; ++k) width += merged_gaps[k];

    // ---- STAGE 3c: thread every row through the merged frame -----------------
    MSA m;
    m.n = n; m.width = width; m.center = center;
    m.rows.assign(n, std::string(width, '-'));

    // The center row: gap columns where merged_gaps demands them, residues else.
    {
        std::string& row = m.rows[center];
        int col = 0;
        for (int k = 0; k < Lc; ++k) {
            col += merged_gaps[k];                  // leading gap columns
            row[col++] = DNA_ALPHABET[s.seq(center)[k]];
        }
        // trailing gaps (merged_gaps[Lc]) stay '-' -- already initialised.
    }

    // Each non-center row: re-thread its pairwise alignment so that its residues
    // land in the same columns as the center, padding with extra gaps where this
    // particular alignment had FEWER center-gaps than the merged maximum.
    for (int b = 0; b < n; ++b) {
        if (b == center) continue;
        std::string& row = m.rows[b];
        const auto& ca = alns[b].a;                 // center row of this pair
        const auto& ba = alns[b].b;                 // the b row of this pair
        int col = 0;                                // current output column
        int k = 0;                                  // next center residue index
        int seen_gaps = 0;                          // center-gaps seen before residue k

        // Helper: when we are about to place center residue k, ensure we have
        // advanced past all merged gap columns for position k (padding row b with
        // '-' for the gap columns it does not itself fill). We pre-pad the
        // difference (merged - this alignment's gaps) as gaps in row b.
        auto pad_to_residue = [&](int kk, int this_gaps) {
            const int extra = merged_gaps[kk] - this_gaps;   // columns to skip as gap
            col += extra;                                    // leave '-' (already set)
        };

        // Walk the pairwise alignment column by column.
        int pending_center_gaps = 0;
        for (std::size_t t = 0; t < ca.size(); ++t) {
            if (ca[t] == GAP_CODE) {
                // A gap in the center: this is an insertion in b. It occupies one
                // of the merged gap columns before center residue k. Place b's
                // residue (ba[t]) into the current column.
                if (ba[t] != GAP_CODE) row[col] = DNA_ALPHABET[ba[t]];
                ++col;
                ++pending_center_gaps;
            } else {
                // A real center residue (index k). First skip any merged gap
                // columns this alignment did NOT use (pad as gaps in b).
                pad_to_residue(k, pending_center_gaps);
                // Now place b's residue (or a gap) aligned to center residue k.
                if (ba[t] != GAP_CODE) row[col] = DNA_ALPHABET[ba[t]];
                ++col;
                ++k;
                seen_gaps += pending_center_gaps;
                pending_center_gaps = 0;
            }
        }
        // Trailing center gaps (after the last center residue): pad as needed.
        pad_to_residue(Lc, pending_center_gaps);
        (void)seen_gaps;                            // (kept for readability)
    }

    m.sp_score = sum_of_pairs(m);
    return m;
}

// ---------------------------------------------------------------------------
// sum_of_pairs: grade an assembled alignment. For each column, for each pair of
// rows (a<b), add the per-residue score: both gaps -> 0; one gap -> NW_GAP; two
// residues -> nw_subst. Integer arithmetic, so it is deterministic and matches
// any other tool using the same scheme. O(width * n^2).
// ---------------------------------------------------------------------------
long long sum_of_pairs(const MSA& m) {
    long long total = 0;
    // Decode a row character back to 0..3, or 4 for a gap.
    auto code = [](char ch) -> uint8_t {
        switch (ch) { case 'A': return 0; case 'C': return 1;
                      case 'G': return 2; case 'T': return 3; default: return 4; }
    };
    for (int col = 0; col < m.width; ++col) {
        for (int a = 0; a < m.n; ++a) {
            const uint8_t ca = code(m.rows[a][col]);
            for (int b = a + 1; b < m.n; ++b) {
                const uint8_t cb = code(m.rows[b][col]);
                if (ca == 4 && cb == 4)        total += 0;        // gap vs gap
                else if (ca == 4 || cb == 4)   total += NW_GAP;   // gap vs residue
                else                            total += nw_subst(ca, cb);
            }
        }
    }
    return total;
}
