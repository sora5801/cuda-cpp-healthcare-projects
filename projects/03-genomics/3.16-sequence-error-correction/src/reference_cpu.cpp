// ===========================================================================
// src/reference_cpu.cpp  --  Trusted serial baseline + dataset loader
// ---------------------------------------------------------------------------
// Project 3.16 : Sequence Error Correction  (k-mer spectrum / trusted-k-mer)
//
// ROLE IN THE PROJECT
//   (1) load_reads()           : parse the tiny FASTA-like sample (data/README.md).
//   (2) build_spectrum_cpu()   : phase-1 reference -- count every k-mer serially.
//   (3) correct_reads_cpu()    : phase-2 reference -- correct every read serially,
//                                using the SHARED correct_one_read() physics from
//                                reference_cpu.h (so the GPU is guaranteed to match).
//   (4) count_residual_errors(): an interpretable science metric (errors left vs
//                                ground truth), not part of CPU==GPU verification.
//
//   Compiled by the host C++ compiler only (no CUDA). Because the per-element
//   logic lives in reference_cpu.h as __host__ __device__ inline functions, this
//   file and kernels.cu call the *same* code -> exact agreement.
//
// READ THIS AFTER: reference_cpu.h. Compare against kernels.cu (the GPU twin).
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

// ---------------------------------------------------------------------------
// load_reads: parse the committed text dataset.
//
//   FILE FORMAT (data/README.md has the canonical spec):
//     line 1            : "<n> <has_truth>"   (n reads; has_truth = 0 or 1)
//     then per read:
//       a "raw" line    : the observed (possibly error-bearing) read, A/C/G/T/N
//       a "truth" line  : the error-free read (ONLY if has_truth==1)
//   Blank lines and lines beginning with '#' are ignored (so the file can carry
//   comments). Reads may have different lengths.
//
//   We store all reads concatenated in `bases` with CSR-style offset/length, the
//   layout the GPU wants (one flat device array). Throws std::runtime_error on a
//   missing file, a bad header, or a truth/raw length mismatch.
// ---------------------------------------------------------------------------
ReadSet load_reads(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open reads file: " + path);

    // Helper: pull the next NON-comment, NON-blank line. Returns false at EOF.
    auto next_line = [&](std::string& out_line) -> bool {
        std::string line;
        while (std::getline(in, line)) {
            // Strip a trailing '\r' so files authored on Windows parse cleanly.
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty() || line[0] == '#') continue;   // skip blanks/comments
            out_line = line;
            return true;
        }
        return false;
    };

    std::string header;
    if (!next_line(header)) throw std::runtime_error("empty reads file: " + path);
    std::istringstream hs(header);
    int n = 0, has_truth = 0;
    if (!(hs >> n >> has_truth))
        throw std::runtime_error("bad header (expected '<n> <has_truth>') in " + path);
    if (n <= 0) throw std::runtime_error("non-positive read count in " + path);

    ReadSet rs;
    rs.n = n;
    rs.has_truth = (has_truth != 0);
    rs.offset.reserve(static_cast<std::size_t>(n) + 1);
    rs.length.reserve(static_cast<std::size_t>(n));
    rs.offset.push_back(0);   // CSR sentinel: read 0 starts at byte 0

    for (int i = 0; i < n; ++i) {
        std::string raw;
        if (!next_line(raw))
            throw std::runtime_error("unexpected EOF reading raw read " +
                                     std::to_string(i) + " in " + path);
        // Append the raw read's bytes to the flat buffer and record its extent.
        rs.bases.insert(rs.bases.end(), raw.begin(), raw.end());
        rs.length.push_back(static_cast<int>(raw.size()));
        rs.offset.push_back(static_cast<int>(rs.bases.size()));

        if (rs.has_truth) {
            std::string truth;
            if (!next_line(truth))
                throw std::runtime_error("unexpected EOF reading truth for read " +
                                         std::to_string(i) + " in " + path);
            if (truth.size() != raw.size())
                throw std::runtime_error("truth/raw length mismatch at read " +
                                         std::to_string(i) + " in " + path);
            rs.truth.insert(rs.truth.end(), truth.begin(), truth.end());
        }
    }
    return rs;
}

// ---------------------------------------------------------------------------
// build_spectrum_cpu: the serial k-mer count (phase-1 reference).
//   Resets `counts` to KMER_TABLE_N zeros, then walks every read and every valid
//   k-mer position, incrementing the slot for that k-mer's code. O(total bases)
//   time. This is the obviously-correct twin of the atomicAdd counting kernel.
// ---------------------------------------------------------------------------
void build_spectrum_cpu(const ReadSet& reads, std::vector<uint32_t>& counts) {
    counts.assign(KMER_TABLE_N, 0u);                 // one slot per possible k-mer
    for (int r = 0; r < reads.n; ++r) {
        const char* seq = &reads.bases[reads.offset[r]];
        const int   len = reads.length[r];
        for (int p = 0; p + KMER_K <= len; ++p) {
            uint32_t code = kmer_code_at(seq, len, p);   // shared encoder
            if (code != 0xFFFFFFFFu) ++counts[code];     // skip windows with 'N'
        }
    }
}

// ---------------------------------------------------------------------------
// correct_reads_cpu: the serial correction pass (phase-2 reference).
//   For each read, copy it into the output buffer and run the SHARED
//   correct_one_read() physics; record how many bases changed. O(reads * len * 4)
//   time. The output bytes are identical to the GPU's because the same inline
//   function decides every substitution.
// ---------------------------------------------------------------------------
void correct_reads_cpu(const ReadSet& reads, const std::vector<uint32_t>& counts,
                       uint32_t thresh, std::vector<char>& corrected,
                       std::vector<int>& changes_per_read) {
    corrected.assign(reads.bases.size(), 0);             // same layout as input
    changes_per_read.assign(static_cast<std::size_t>(reads.n), 0);
    for (int r = 0; r < reads.n; ++r) {
        const char* in  = &reads.bases[reads.offset[r]];
        char*       out = &corrected[reads.offset[r]];
        const int   len = reads.length[r];
        changes_per_read[static_cast<std::size_t>(r)] =
            correct_one_read(in, out, len, counts.data(), thresh);
    }
}

// ---------------------------------------------------------------------------
// count_residual_errors: how many corrected bases still differ from the truth.
//   Only meaningful for synthetic data where we KNOW the error-free read. This
//   is a SCIENCE metric (did correction actually reduce errors?) and is reported
//   on stdout, but it is independent of the CPU==GPU agreement check.
//   Returns -1 if no ground truth is available.
// ---------------------------------------------------------------------------
long count_residual_errors(const ReadSet& reads, const std::vector<char>& corrected) {
    if (!reads.has_truth || reads.truth.empty()) return -1;
    long mismatches = 0;
    for (int r = 0; r < reads.n; ++r) {
        const int beg = reads.offset[r];
        const int len = reads.length[r];
        for (int i = 0; i < len; ++i)
            if (corrected[beg + i] != reads.truth[beg + i]) ++mismatches;
    }
    return mismatches;
}
