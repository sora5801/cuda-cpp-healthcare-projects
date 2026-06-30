// ===========================================================================
// src/reference_cpu.cpp  --  Trusted serial profile-HMM scorers + data loader
// ---------------------------------------------------------------------------
// Project 3.28 : Profile HMM (Viterbi / Forward)
//
// ROLE
//   (1) aa_code / load_database()        : parse the tiny FASTA-like dataset.
//   (2) build_profile_from_consensus()   : turn a consensus string into a model.
//   (3) viterbi_cpu() / forward_cpu()    : the obviously-correct serial 2-D DP
//       scorers the GPU kernels are verified against. They loop the SAME per-cell
//       recurrences (phmm.h) the GPU thread loops, so the two agree to machine
//       precision (THEORY §6).
//
//   Compiled by the host C++ compiler only (no CUDA). See reference_cpu.h.
//
// READ THIS AFTER: reference_cpu.h, phmm.h.  Compare against kernels.cu (the GPU
//   twin that runs the identical recurrence, one thread per sequence).
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

// ===========================================================================
// 1. ALPHABET + LOADER
// ===========================================================================

// The 20 standard amino acids in a fixed canonical order. The INDEX of a letter
// in this string IS its integer code (so 'A'->0, 'C'->1, ... 'Y'->19). Keeping
// one canonical ordering here means the loader and the model builder can never
// disagree about what "residue 7" means.
static const char* AA_ORDER = "ACDEFGHIKLMNPQRSTVWY";

int aa_code(char c) {
    // Uppercase the input so lowercase FASTA still parses.
    if (c >= 'a' && c <= 'z') c = static_cast<char>(c - 'a' + 'A');
    for (int i = 0; i < ALPHA; ++i)
        if (AA_ORDER[i] == c) return i;
    return -1;   // not one of the 20 standard amino acids
}

SeqDB load_database(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open database file: " + path);

    // FASTA-like format (see data/README.md):
    //   lines beginning with '>' start a new record; the token after '>' is the
    //   record name. Following non-'>' lines are residues for the current record.
    //   We concatenate residue codes for each record into the flat SeqDB.
    SeqDB db;
    std::string cur_name;
    std::string cur_seq;

    // flush(): when a record ends, validate it and append it to the SeqDB.
    auto flush = [&]() {
        if (cur_name.empty() && cur_seq.empty()) return;   // nothing pending
        if (cur_seq.empty())
            throw std::runtime_error("record '" + cur_name + "' has no residues in " + path);
        if (static_cast<int>(cur_seq.size()) > MAX_L)
            throw std::runtime_error("sequence '" + cur_name + "' length " +
                                     std::to_string(cur_seq.size()) +
                                     " exceeds MAX_L=" + std::to_string(MAX_L));
        db.off.push_back(static_cast<int>(db.res.size()));   // where this seq starts
        db.len.push_back(static_cast<int>(cur_seq.size()));
        db.name.push_back(cur_name);
        for (char c : cur_seq) {
            int a = aa_code(c);
            if (a < 0) throw std::runtime_error(std::string("unknown residue '") + c +
                                                "' in record '" + cur_name + "'");
            db.res.push_back(static_cast<std::uint8_t>(a));
        }
        ++db.n;
        cur_name.clear();
        cur_seq.clear();
    };

    std::string line;
    while (std::getline(in, line)) {
        // Strip a trailing '\r' so Windows-edited files parse on any platform.
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;                 // blank lines are ignored
        if (line[0] == '#') continue;               // '#' lines are comments
        if (line[0] == '>') {                       // header -> start a new record
            flush();                                // finish the previous one
            std::istringstream hs(line.substr(1));  // text after '>'
            hs >> cur_name;                         // first token is the name
            if (cur_name.empty()) cur_name = "seq" + std::to_string(db.n);
        } else {
            // Residue line: append (ignoring internal spaces, which some FASTA
            // pretty-printers insert for readability).
            for (char c : line)
                if (c != ' ' && c != '\t') cur_seq.push_back(c);
        }
    }
    flush();   // don't forget the final record (no trailing '>' to trigger it)

    if (db.n == 0) throw std::runtime_error("no sequences found in " + path);
    return db;
}

// ===========================================================================
// 2. MODEL BUILDER
// ---------------------------------------------------------------------------
// build_profile_from_consensus(): construct a ProfileHMM whose match columns
// favor the consensus residues. This is the teaching stand-in for "fit a profile
// to a family alignment". The numbers below are chosen to be realistic in spirit
// (a strong but not absolute preference for the consensus residue) and FIXED so
// the demo is deterministic.
// ===========================================================================
ProfileHMM build_profile_from_consensus(const std::string& consensus) {
    const int M = static_cast<int>(consensus.size());
    if (M < 1 || M > MAX_M)
        throw std::runtime_error("consensus length " + std::to_string(M) +
                                 " must be in 1.." + std::to_string(MAX_M));

    ProfileHMM p{};            // value-init -> all doubles 0.0 (we overwrite below)
    p.M = M;

    // --- Emission probabilities --------------------------------------------
    // Each match column k emits its consensus residue with probability P_HIT and
    // splits the remaining mass over the other 19 residues. We store NATURAL LOGS
    // (the DP works in log space; see phmm.h). These two constants set how
    // "peaked" the family signal is -- a sharper peak makes homologs score much
    // higher than random sequences (the planted-answer effect, PATTERNS.md §6).
    const double P_HIT  = 0.55;                        // P(consensus residue | match k)
    const double P_MISS = (1.0 - P_HIT) / (ALPHA - 1); // P(any other residue | match k)
    const double LOG_HIT  = std::log(P_HIT);
    const double LOG_MISS = std::log(P_MISS);

    for (int k = 1; k <= M; ++k) {
        const int consensus_code = aa_code(consensus[k - 1]);
        if (consensus_code < 0)
            throw std::runtime_error(std::string("bad consensus residue '") +
                                     consensus[k - 1] + "'");
        for (int a = 0; a < ALPHA; ++a)
            p.match_emit[k * ALPHA + a] = (a == consensus_code) ? LOG_HIT : LOG_MISS;
    }
    // Insert states emit from the BACKGROUND (uniform over 20 residues here),
    // matching HMMER's convention that insert emissions ~ background, so an
    // insertion carries no family signal -- only match columns do.
    const double LOG_BG = std::log(1.0 / ALPHA);
    for (int a = 0; a < ALPHA; ++a) p.insert_emit[a] = LOG_BG;

    // --- Transition probabilities ------------------------------------------
    // Plan-7-style defaults: staying on the match track is the overwhelmingly
    // likely path; insertions and deletions are rare but possible. These are the
    // same per column (a homogeneous model) for teaching clarity. Each group must
    // be a probability distribution (sums to 1) BEFORE we take logs.
    //   From M: mostly M->M, small M->I and M->D.
    //   From I: mostly I->M, some I->I (geometric insert length).
    //   From D: mostly D->M, some D->D (geometric delete length).
    const double mm = 0.90, mi = 0.05, md = 0.05;   // sums to 1.0
    const double im = 0.70, ii = 0.30;              // sums to 1.0
    const double dm = 0.70, dd = 0.30;              // sums to 1.0
    TransLog t;
    t.mm = std::log(mm); t.mi = std::log(mi); t.md = std::log(md);
    t.im = std::log(im); t.ii = std::log(ii);
    t.dm = std::log(dm); t.dd = std::log(dd);
    for (int k = 0; k <= M; ++k) p.trans[k] = t;    // homogeneous: same every column

    return p;
}

// ===========================================================================
// 3. THE SERIAL DP SCORERS
// ---------------------------------------------------------------------------
// score_one(): the SHARED inner routine. It fills the M/I/D dynamic-programming
// lattice for ONE sequence and returns the final score. A template parameter
// `IS_VITERBI` selects the combine operator at COMPILE TIME so there is no
// per-cell branch:
//   * IS_VITERBI == true  -> combine = max  (Viterbi: best single path)
//   * IS_VITERBI == false -> combine = log_sum_exp (Forward: sum over all paths)
// Both call the exact phmm.h helpers the GPU calls, so CPU and GPU match.
//
// LATTICE LAYOUT (one sequence of length L against a profile of length M):
//   We keep only TWO rows of each plane at a time (the "previous" row i-1 and the
//   "current" row i) because the M/I recurrence only reaches back one residue.
//   This is the same rolling-row trick the GPU thread uses, and it bounds memory
//   to O(M) instead of O(L*M). Index k runs 0..M (column 0 is the boundary).
//
//   Planes (each a length-(M+1) row, doubles, in log space):
//     Mrow / Irow / Drow      -> current row i
//     Mprev / Iprev / Dprev   -> previous row i-1
//
//   BOUNDARY / BEGIN: before any residue is emitted (i=0) the only reachable
//   states are the "begin" path. We seed M[0][1] = 0 (log 1) as the entry point
//   and let a silent DELETE chain walk across row 0 so a path may skip leading
//   columns. END reads the score from the final match column M after the last
//   residue. This is a deliberately simplified begin/end -- THEORY §7 explains
//   the full Plan-7 N/B/E/C/J flanking-state machinery we omit.
// ===========================================================================
template <bool IS_VITERBI>
static double score_one(const ProfileHMM& p, const std::uint8_t* seq, int L) {
    const int M = p.M;

    // combine(): the ONE operator that differs between the two algorithms. The
    // ternary on a compile-time-constant template arg is folded away by the
    // optimizer, so there is no runtime branch in the inner loop.
    auto combine = [](double a, double b) -> double {
        return IS_VITERBI ? max2(a, b) : log_sum_exp(a, b);
    };
    auto combine3 = [](double a, double b, double c) -> double {
        return IS_VITERBI ? max3(a, b, c) : log_sum_exp(log_sum_exp(a, b), c);
    };

    // Rolling rows. Initialize to LOG_ZERO ("impossible") everywhere.
    std::vector<double> Mprev(M + 1, LOG_ZERO), Iprev(M + 1, LOG_ZERO), Dprev(M + 1, LOG_ZERO);
    std::vector<double> Mrow(M + 1, LOG_ZERO),  Irow(M + 1, LOG_ZERO),  Drow(M + 1, LOG_ZERO);

    // ---- Row i = 0 : the BEGIN boundary (no residue emitted yet) -----------
    // A path begins by entering match column 1 directly (M[0][1] = log 1 = 0),
    // then may walk a silent chain of DELETE states to skip leading columns
    // before emitting anything. We do not allow inserts before the first residue.
    Mprev[1] = 0.0;   // "begin -> M_1" with log-prob 0 (probability 1)
    for (int k = 2; k <= M; ++k) {
        const TransLog& tk1 = p.trans[k - 1];
        double from_m = Mprev[k - 1] + tk1.md;    // M_{k-1} -> D_k
        double from_d = Dprev[k - 1] + tk1.dd;    // D_{k-1} -> D_k
        Dprev[k] = combine(from_m, from_d);
    }

    // ---- Rows i = 1..L : emit residue x_i ----------------------------------
    for (int i = 1; i <= L; ++i) {
        const int x = seq[i - 1];   // residue code emitted at this row

        // Reset the current row to "impossible"; we will fill it in.
        for (int k = 0; k <= M; ++k) { Mrow[k] = Irow[k] = Drow[k] = LOG_ZERO; }

        for (int k = 1; k <= M; ++k) {
            // MATCH M[i][k]: emit x from match column k, arriving from M/I/D at
            // (i-1, k-1). Add the match emission AFTER combining the incoming
            // transitions (emission is independent of which predecessor we came
            // from). trans[k-1] holds the X->Y(column k) transition logs.
            const TransLog& tk1 = p.trans[k - 1];
            double in_m = Mprev[k - 1] + tk1.mm;    // M_{k-1} -> M_k
            double in_i = Iprev[k - 1] + tk1.im;    // I_{k-1} -> M_k
            double in_d = Dprev[k - 1] + tk1.dm;    // D_{k-1} -> M_k
            Mrow[k] = combine3(in_m, in_i, in_d) + emit_match(p, k, x);

            // INSERT I[i][k]: emit x from insert state k, arriving from M/I at
            // (i-1, k). trans[k] holds the X(column k)->I_k transition logs.
            const TransLog& tk = p.trans[k];
            double in2_m = Mprev[k] + tk.mi;       // M_k -> I_k
            double in2_i = Iprev[k] + tk.ii;        // I_k -> I_k
            Irow[k] = combine(in2_m, in2_i) + emit_insert(p, x);

            // DELETE D[i][k]: a SILENT state (emits nothing), so it reads from the
            // CURRENT row at column k-1 (same i). We compute k left-to-right so
            // Drow[k-1] is already final when we use it.
            double in3_m = Mrow[k - 1] + tk1.md;    // M_{k-1} -> D_k  (current row)
            double in3_d = Drow[k - 1] + tk1.dd;    // D_{k-1} -> D_k  (current row)
            Drow[k] = combine(in3_m, in3_d);
        }

        // Roll: the current row becomes "previous" for the next residue. Three
        // independent swaps (O(1) each, no copies).
        Mprev.swap(Mrow);
        Iprev.swap(Irow);
        Dprev.swap(Drow);
    }

    // ---- END: read out the score ------------------------------------------
    // The alignment must END in the final match column M after consuming all L
    // residues. After the final swap, the last emitted row lives in *prev.
    return Mprev[M];
}

// Public wrappers: loop score_one over the database, choosing the algorithm.
void viterbi_cpu(const ProfileHMM& p, const SeqDB& db, std::vector<float>& out) {
    out.assign(static_cast<std::size_t>(db.n), 0.0f);
    for (int s = 0; s < db.n; ++s) {
        const std::uint8_t* seq = &db.res[db.off[s]];
        out[s] = static_cast<float>(score_one<true>(p, seq, db.len[s]));
    }
}

void forward_cpu(const ProfileHMM& p, const SeqDB& db, std::vector<float>& out) {
    out.assign(static_cast<std::size_t>(db.n), 0.0f);
    for (int s = 0; s < db.n; ++s) {
        const std::uint8_t* seq = &db.res[db.off[s]];
        out[s] = static_cast<float>(score_one<false>(p, seq, db.len[s]));
    }
}
