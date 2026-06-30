// ===========================================================================
// src/ctc_core.h  --  The ONE TRUE per-read CTC decode, shared by CPU and GPU
// ---------------------------------------------------------------------------
// Project 3.4 : Nanopore Basecalling  (REDUCED-SCOPE TEACHING VERSION)
//
// WHY THIS HEADER EXISTS  (PATTERNS.md sec 2: the __host__ __device__ core)
//   The single most useful idiom in this repo: put the per-element MATH in one
//   header marked `__host__ __device__` so the CPU reference (reference_cpu.cpp,
//   compiled by cl.exe/g++) and the GPU kernel (kernels.cu, compiled by nvcc)
//   run BYTE-FOR-BYTE IDENTICAL code. That makes verification EXACT (tolerance
//   == 0) instead of approximate: if the two sides disagree even by one base,
//   it is a real bug, not floating-point noise.
//
//   Keep CUDA-only things OUT of this header (no __global__, no <<<>>>), so the
//   host compiler can include it happily. Only the HD decorator macro is CUDA-
//   aware, and it vanishes to nothing under the host compiler.
//
// WHAT THIS PROJECT ACTUALLY TEACHES
//   Real nanopore basecalling (Oxford Nanopore's Dorado/Guppy) is a two-stage
//   pipeline:
//       raw current squiggle  --[ neural network: LSTM/transformer ]-->  a
//       posterior matrix P of shape [T x C]  --[ CTC decode ]-->  DNA bases.
//   The NEURAL NETWORK is research-grade (trained weights, cuDNN/TensorRT) and
//   is explicitly OUT OF SCOPE here (CLAUDE.md sec 13: ship the tractable
//   teaching slice, describe the full thing in THEORY). We implement the SECOND
//   stage -- the CTC GREEDY DECODE -- which is the step that turns the network's
//   output probabilities into an actual base sequence. It is:
//       * the part a learner can fully understand and verify,
//       * embarrassingly parallel ACROSS READS (one read per GPU thread), and
//       * the exact algorithm Guppy's "fast" mode and Dorado's greedy path use.
//
// THE CTC ALPHABET
//   A nanopore network emits, at each of T time steps, a probability over C = 5
//   classes: a BLANK symbol plus the four DNA bases. We use this fixed order:
//       index 0 = blank ('-'),  1 = A,  2 = C,  3 = G,  4 = T
//   "Blank" is CTC's way of saying "no new base emitted at this step"; it is how
//   the model represents both gaps between bases and the duration a base dwells
//   in the pore (many consecutive steps for one physical base).
//
// THE GREEDY-DECODE RULE (a.k.a. "best path" / argmax collapse)
//   1. ARGMAX: at each time step t, pick the most probable class a_t.
//   2. COLLAPSE: walk the argmax path a_0..a_{T-1} and
//        (a) merge runs of the SAME class (a base that dwells many steps -> 1),
//        (b) then delete all BLANKs.
//   The surviving symbols, in order, are the called bases. Example path
//   (blank='-'):  - A A - A C C  ->  (merge) - A - A C  ->  (drop blank) A A C.
//   Note that "A A" survives because the blank between the two A-runs separates
//   them -- that is the whole point of the blank symbol.
//
// READ THIS BEFORE: reference_cpu.cpp (loops this over reads on the CPU) and
//   kernels.cu (calls this from one GPU thread per read). The science/math is in
//   ../THEORY.md.
// ===========================================================================
#pragma once

#include <cstdint>

// ---------------------------------------------------------------------------
// HD: the host/device decorator.
//   * Compiled by nvcc (__CUDACC__ defined): expands to `__host__ __device__`
//     so the SAME function is emitted for both the CPU and the GPU.
//   * Compiled by the host C++ compiler: expands to nothing (those keywords do
//     not exist there). reference_cpu.cpp then sees plain inline functions.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// CTC_UNROLL: `#pragma unroll` is an nvcc-only hint. Under the host compiler
// (cl.exe / g++) it is an UNKNOWN pragma and triggers a warning (MSVC C4068),
// so we expand it to nothing there. This keeps reference_cpu.cpp warning-clean
// while still asking nvcc to unroll the short, compile-time-bounded loops.
#ifdef __CUDACC__
#define CTC_UNROLL _Pragma("unroll")
#else
#define CTC_UNROLL
#endif

// The CTC alphabet. C = number of classes per time step. Fixed at compile time
// so the inner argmax loop unrolls and the layout is simple. blank == 0.
constexpr int CTC_NUM_CLASSES = 5;   // {blank, A, C, G, T}
constexpr int CTC_BLANK       = 0;   // class index of the blank symbol

// Map a (non-blank) class index 1..4 to its DNA letter. Class 0 (blank) never
// reaches output, so we return 'N' for it as a defensive sentinel. Kept here
// (HD) so the GPU can build the same ASCII string the CPU does.
HD inline char ctc_class_to_base(int cls) {
    // A tiny lookup table beats a switch for branch-free, identical CPU/GPU code.
    // Index by class: [blank, A, C, G, T].
    const char table[CTC_NUM_CLASSES] = {'N', 'A', 'C', 'G', 'T'};
    if (cls < 0 || cls >= CTC_NUM_CLASSES) return 'N';   // guard out-of-range
    return table[cls];
}

// ---------------------------------------------------------------------------
// ctc_argmax_step: index of the most probable class at one time step.
//   p   : pointer to this step's C probabilities (p[0..C-1]); read-only.
//   Ties are broken by the LOWEST class index (we only replace the running best
//   on a STRICTLY greater probability). This makes the decode DETERMINISTIC and
//   identical on CPU and GPU -- crucial for the exact (tolerance==0) check.
//
//   Why argmax and not a softmax/threshold? Greedy CTC only needs the most
//   likely class per step; the probabilities' absolute scale is irrelevant, so
//   this works whether p holds raw logits, probabilities, or log-probs (argmax
//   is monot-invariant). Our synthetic data stores probabilities for clarity.
// ---------------------------------------------------------------------------
HD inline int ctc_argmax_step(const float* p) {
    int   best_cls = 0;          // running best class index (start at class 0)
    float best_val = p[0];       // its probability
    // Compile-time-bounded loop -> fully unrolled; just 4 compare-selects.
    CTC_UNROLL
    for (int c = 1; c < CTC_NUM_CLASSES; ++c) {
        // STRICT '>' so equal probabilities keep the lower index (determinism).
        if (p[c] > best_val) { best_val = p[c]; best_cls = c; }
    }
    return best_cls;
}

// ---------------------------------------------------------------------------
// ctc_greedy_decode: decode ONE read's posterior matrix into a base string.
//   probs    : [T * C] row-major posteriors for this read. Row t is the C class
//              probabilities at time step t; element (t,c) = probs[t*C + c].
//   T        : number of time steps (rows) for this read.
//   out_bases: caller-provided buffer of at least T chars; receives the called
//              bases (NOT null-terminated here -- caller knows the length).
//   returns  : the number of bases written (the decoded read length, <= T).
//
//   This is the heart of the project. It performs argmax + CTC collapse in a
//   SINGLE forward pass over the T steps, tracking only the previous argmax
//   class so it can (a) merge repeats and (b) drop blanks on the fly -- O(T*C)
//   work, O(1) extra state. One GPU thread runs this for one read; the CPU
//   reference runs it in a host loop over reads. Same code, same answer.
//
//   Determinism: every operation is an integer compare or an array write in a
//   fixed order, so the output is bit-identical across runs and across CPU/GPU.
// ---------------------------------------------------------------------------
HD inline int ctc_greedy_decode(const float* probs, int T, char* out_bases) {
    int  n_out    = 0;            // how many bases emitted so far
    int  prev_cls = -1;          // argmax class of the PREVIOUS step (-1 = none)
    for (int t = 0; t < T; ++t) {
        // Row t starts at probs[t*C]; pick its most probable class.
        const int cls = ctc_argmax_step(probs + static_cast<long long>(t) * CTC_NUM_CLASSES);

        // CTC COLLAPSE, done incrementally:
        //   * skip if this class repeats the immediately previous one (merge a
        //     base that dwells across several steps into a single call), and
        //   * skip blanks entirely (they are never emitted).
        //   A base is emitted only when the class CHANGES to a NON-blank value.
        if (cls != prev_cls && cls != CTC_BLANK) {
            out_bases[n_out++] = ctc_class_to_base(cls);
        }
        // Advance the "previous" marker. Note we set it to the raw argmax class
        // (including blank), so e.g. "A A blank A" decodes to "A A": the blank
        // resets prev_cls, letting the second A-run emit a fresh base.
        prev_cls = cls;
    }
    return n_out;
}

// ---------------------------------------------------------------------------
// ctc_base_checksum: a tiny order-sensitive integer fingerprint of a decoded
//   read, used as a DETERMINISTIC, compact correctness signal in stdout.
//   bases : pointer to n called bases (chars 'A'/'C'/'G'/'T').
//   n     : number of bases.
//   returns: a 32-bit rolling hash. Integer-only (adds/multiplies in a fixed
//            order) so it is reproducible and identical on CPU and GPU -- we do
//            NOT use floats here precisely so the value is exact (PATTERNS.md
//            sec 3: integer reductions are deterministic; float sums are not).
//
//   This is the classic Java-string-style polynomial hash h = h*31 + ch. It is
//   NOT cryptographic; its only job is to let the demo print one stable number
//   per read that changes if a single base changes -- a human-checkable proof
//   that CPU and GPU produced the identical sequence.
// ---------------------------------------------------------------------------
HD inline uint32_t ctc_base_checksum(const char* bases, int n) {
    uint32_t h = 2166136261u;     // a fixed nonzero seed (FNV offset basis)
    for (int i = 0; i < n; ++i) {
        // Cast through unsigned char so the arithmetic is well-defined and
        // identical regardless of whether `char` is signed on this platform.
        h = h * 31u + static_cast<unsigned char>(bases[i]);
    }
    return h;
}
