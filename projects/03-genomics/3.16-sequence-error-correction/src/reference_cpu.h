// ===========================================================================
// src/reference_cpu.h  --  Data model + shared k-mer correction "physics"
// ---------------------------------------------------------------------------
// Project 3.16 : Sequence Error Correction  (k-mer spectrum / trusted-k-mer)
//
// WHY THIS HEADER IS THE CENTRE OF THE PROJECT
//   This single header is included by BOTH compilers:
//     * the host C++ compiler, when it builds reference_cpu.cpp and main.cu's
//       host portions, and
//     * nvcc, when it builds kernels.cu.
//   Everything that decides "is this k-mer trusted?" and "how do we correct a
//   base?" lives here as `__host__ __device__` inline functions (the HD-macro
//   idiom, PATTERNS.md sec 2). The CPU reference and the GPU kernel therefore run
//   BYTE-FOR-BYTE IDENTICAL integer logic, so verification is *exact* (==), not a
//   floating-point tolerance. (No CUDA-only types appear here -- no `__global__`,
//   no `dim3` -- so the plain host compiler can include it happily.)
//
// THE PROBLEM (full derivation in ../THEORY.md)
//   A DNA sequencer reads the genome by sampling many short overlapping "reads".
//   Each base call has a small error probability, so reads contain wrong bases.
//   Error CORRECTION fixes those wrong bases *before* assembly. The dominant
//   short-read method is the K-MER SPECTRUM:
//     * Slide a length-k window over every read to enumerate its k-mers.
//     * Count how many times each distinct k-mer occurs across ALL reads (the
//       "spectrum"). A k-mer from the true genome is seen on many overlapping
//       reads -> HIGH count; a k-mer containing a sequencing error is essentially
//       random -> count of 1 or 2.
//     * Pick a coverage threshold T. k-mers with count >= T are "TRUSTED"
//       (believed real); the rest are "untrusted" (likely error-bearing).
//     * To correct a read: a base covered only by untrusted k-mers is suspect.
//       Try substituting each of the other 3 bases; if a substitution turns the
//       k-mers spanning that base from untrusted into trusted, accept it.
//
//   Two GPU phases, both embarrassingly parallel (see kernels.cuh):
//     Phase 1 (count): one thread per k-mer occurrence, atomicAdd into a table.
//     Phase 2 (correct): one thread per read, fully independent.
//
// READ THIS BEFORE: reference_cpu.cpp, kernels.cuh.
// ===========================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// The HD macro: decorate the shared functions so nvcc compiles them for BOTH
// the host and the device, while the plain host compiler (which has never heard
// of `__host__`/`__device__`) sees nothing. This is what guarantees the CPU and
// GPU run the same arithmetic. __CUDACC__ is defined only when nvcc is compiling.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define SEC_HD __host__ __device__
#else
#define SEC_HD
#endif

// ---------------------------------------------------------------------------
// K-MER GEOMETRY (compile-time constants so loops unroll and tables size at
// compile time). We use k = 9 for the teaching version:
//   * There are 4^9 = 262,144 possible 9-mers over the DNA alphabet {A,C,G,T}.
//   * We DIRECT-INDEX a count table of that many uint32 slots (~1 MB) -- one
//     slot per possible k-mer -- so there are NO hash collisions to reason about.
//     Production tools (CARE/BFC) cannot do this for k=21..31 (4^31 slots is
//     astronomically large), so they use a real GPU HASH TABLE with atomic CAS;
//     we explain that trade-off in THEORY sec 7 and keep the exact table here
//     because it makes the spectrum idea perfectly legible.
// ---------------------------------------------------------------------------
constexpr int      KMER_K       = 9;                       // k-mer length (bases)
constexpr uint32_t KMER_TABLE_N = 1u << (2 * KMER_K);      // 4^9 = 262144 slots
constexpr uint32_t KMER_MASK    = KMER_TABLE_N - 1u;       // low 2*k bits = a k-mer code

// Encode the 4 DNA bases as 2-bit numbers. Any non-ACGT base (e.g. 'N') maps to
// a sentinel so we can SKIP k-mers that contain it (they are uncountable).
//   A=0, C=1, G=2, T=3 ; anything else -> 0xFF ("not a base").
SEC_HD inline uint8_t base_code(char c) {
    switch (c) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'T': case 't': return 3;
        default:            return 0xFF;   // 'N' or junk -> sentinel
    }
}

// Inverse map: a 2-bit code back to its uppercase base letter (for writing the
// corrected read out). Index must be 0..3.
SEC_HD inline char code_base(uint8_t code) {
    // A tiny lookup; constant so it compiles to a single indexed load.
    const char LUT[4] = {'A', 'C', 'G', 'T'};
    return LUT[code & 3u];
}

// ---------------------------------------------------------------------------
// kmer_code_at: encode the length-k window of `seq` starting at offset `pos`
// into its integer code (base i contributes 2 bits at position 2*(k-1-i), i.e.
// the first base is the most-significant pair -- a fixed, canonical order shared
// by CPU and GPU). Returns 0xFFFFFFFF if the window runs off the end OR contains
// a non-ACGT base, signalling "no valid k-mer here".
//   seq : pointer to the read's bytes (ASCII A/C/G/T/N)
//   len : length of the read
//   pos : start offset of the window (0 <= pos, window is [pos, pos+k))
// ---------------------------------------------------------------------------
SEC_HD inline uint32_t kmer_code_at(const char* seq, int len, int pos) {
    if (pos < 0 || pos + KMER_K > len) return 0xFFFFFFFFu;   // window off the end
    uint32_t code = 0;
    for (int i = 0; i < KMER_K; ++i) {
        uint8_t b = base_code(seq[pos + i]);
        if (b == 0xFF) return 0xFFFFFFFFu;                   // contains 'N'/junk
        code = (code << 2) | b;                              // shift in 2 bits
    }
    return code & KMER_MASK;
}

// is_trusted: a k-mer code is "trusted" iff its spectrum count is at least the
// coverage threshold T. This one predicate is the heart of the method; both the
// CPU reference and the GPU kernel call it so they agree by construction.
//   counts : the spectrum table (KMER_TABLE_N uint32 slots)
//   code   : a valid k-mer code (caller guarantees code < KMER_TABLE_N)
//   thresh : trust threshold T (counts >= T  =>  trusted)
SEC_HD inline bool is_trusted(const uint32_t* counts, uint32_t code, uint32_t thresh) {
    return counts[code] >= thresh;
}

// ---------------------------------------------------------------------------
// correct_one_read: THE shared per-read correction routine (the "physics").
//
//   The teaching algorithm (a deliberately simple, deterministic flavour of the
//   trusted-k-mer correction used by CARE/Quake/BFC):
//     For each base position p that starts a k-mer, from left to right:
//       1. If the k-mer STARTING at p exists and is already trusted, leave it.
//       2. Otherwise this base may be an error. Try each alternative base
//          (A,C,G,T in fixed order, skipping the current one). A candidate base
//          is ACCEPTED if it makes the k-mer starting at p trusted. Among
//          accepted candidates we keep the one whose new k-mer has the HIGHEST
//          spectrum count (ties broken by A,C,G,T base order) -- a fixed rule, so
//          the result is deterministic and identical on CPU and GPU.
//       3. At most ONE substitution per position; we make a single left-to-right
//          pass. This corrects the common case (isolated substitution errors)
//          while staying O(read length) and trivially parallel across reads.
//
//   This writes the corrected read in place into `out` (a private copy the
//   caller owns) and RETURNS the number of bases it changed. Because every
//   decision is an integer comparison against the shared `counts` table, the
//   CPU loop and the GPU thread produce identical output.
//
//   in     : the raw read bytes (length `len`), ASCII A/C/G/T/N
//   out    : caller-provided buffer of >= len bytes; receives the corrected read
//   len    : read length
//   counts : spectrum table (KMER_TABLE_N slots)
//   thresh : trust threshold T
//   returns: number of substituted bases (>= 0)
// ---------------------------------------------------------------------------
SEC_HD inline int correct_one_read(const char* in, char* out, int len,
                                   const uint32_t* counts, uint32_t thresh) {
    // Start from an exact copy; we edit `out` so earlier corrections feed later
    // k-mers (a corrected base helps the next overlapping window). `in` is never
    // modified -- the CPU and GPU both treat the input as read-only.
    for (int i = 0; i < len; ++i) out[i] = in[i];

    int changes = 0;
    // Walk every position that can START a k-mer. The last valid start is
    // len-KMER_K; positions beyond that have no full k-mer to anchor on.
    for (int p = 0; p + KMER_K <= len; ++p) {
        uint32_t code = kmer_code_at(out, len, p);
        if (code == 0xFFFFFFFFu) continue;            // 'N' in window -> skip
        if (is_trusted(counts, code, thresh)) continue;  // already good, leave it

        // This window is untrusted. The most likely single culprit under a
        // substitution-error model is the FIRST base of the window (position p):
        // p is the left-most base we have not yet locked in, and an error there
        // makes this window look wrong. Try the 3 other bases at p and keep the
        // one giving the strongest trusted k-mer.
        char    orig      = out[p];
        uint8_t orig_code = base_code(orig);
        if (orig_code == 0xFF) continue;              // can't reason about 'N'

        uint8_t  best_base  = orig_code;              // default: no change
        uint32_t best_count = 0;                      // best spectrum count seen
        bool     found      = false;

        // Fixed A,C,G,T order -> deterministic tie-breaking on both CPU and GPU.
        for (uint8_t cand = 0; cand < 4; ++cand) {
            if (cand == orig_code) continue;          // skip the current base
            out[p] = code_base(cand);                 // tentatively substitute
            uint32_t new_code = kmer_code_at(out, len, p);
            if (new_code != 0xFFFFFFFFu &&
                is_trusted(counts, new_code, thresh) &&
                counts[new_code] > best_count) {
                best_count = counts[new_code];        // remember the strongest
                best_base  = cand;
                found      = true;
            }
        }

        // Commit the best accepted substitution (or restore the original if none
        // of the alternatives produced a trusted k-mer -- we never guess blindly).
        out[p] = code_base(found ? best_base : orig_code);
        if (found && best_base != orig_code) ++changes;
    }
    return changes;
}

// ===========================================================================
// DATA MODEL  (pure C++, shared by host + device code that loads/holds reads)
// ===========================================================================

// A loaded read set. Reads can have different lengths, so we store all read
// bytes CONCATENATED in one flat buffer plus per-read (offset,length) so the
// layout is GPU-friendly (one contiguous device array, no pointer-chasing).
//   bases   : all reads' bytes concatenated, ASCII A/C/G/T/N
//   offset  : offset[i] = start of read i inside `bases`  (size n+1; offset[n]
//             = total length, a standard CSR-style sentinel)
//   length  : length[i] = number of bases in read i
//   n       : number of reads
//   truth   : OPTIONAL ground-truth (error-free) reads, same layout as `bases`
//             via offset/length; present only for SYNTHETIC data so we can score
//             how many errors remain. Empty if unknown.
struct ReadSet {
    int                  n = 0;
    std::vector<char>    bases;     // concatenated read bytes
    std::vector<int>     offset;    // size n+1, CSR-style
    std::vector<int>     length;    // size n
    std::vector<char>    truth;     // concatenated ground-truth bytes (or empty)
    bool has_truth = false;         // true iff `truth` is populated
};

// Parse the tiny FASTA-like text dataset documented in data/README.md. Throws
// std::runtime_error on a missing file or malformed header.
ReadSet load_reads(const std::string& path);

// CPU reference, phase 1: build the spectrum (count every valid k-mer over all
// reads) into `counts` (resized to KMER_TABLE_N). Pure serial loop -- the
// obviously-correct baseline the GPU counting kernel is verified against.
void build_spectrum_cpu(const ReadSet& reads, std::vector<uint32_t>& counts);

// CPU reference, phase 2: correct every read using the shared correct_one_read()
// "physics". Fills `corrected` (concatenated, same offset/length layout as the
// input) and `changes_per_read` (size n). This is the trusted baseline.
void correct_reads_cpu(const ReadSet& reads, const std::vector<uint32_t>& counts,
                       uint32_t thresh, std::vector<char>& corrected,
                       std::vector<int>& changes_per_read);

// Count how many bases of `corrected` still disagree with the ground truth, over
// all reads. Returns -1 if no truth is available. Used only as an interpretable
// SCIENCE metric (errors-before vs errors-after); not part of CPU==GPU checking.
long count_residual_errors(const ReadSet& reads, const std::vector<char>& corrected);
