// ===========================================================================
// src/generator.h  --  Shared (host + device) generative model + scorer
// ---------------------------------------------------------------------------
// Project 1.10 : De Novo Generative Molecular Design  (REDUCED-SCOPE teaching
//                version -- see ../THEORY.md "Where this sits in the real world"
//                for how production tools like REINVENT4 differ).
//
// WHY THIS HEADER IS SHARED (the single most important idea in this project)
//   De-novo design has three steps: (1) a *generative model* defines a
//   probability over the next token given the context so far; (2) we *sample*
//   from it autoregressively to emit a novel molecule string; (3) we *score*
//   each molecule and keep the best (goal-directed optimisation). To VERIFY a
//   GPU implementation we run the EXACT same three steps on the CPU and demand
//   bit-identical molecules and scores. That only works if both sides share one
//   RNG, one sampling loop, and one scorer -- so all of it lives HERE, in a
//   single header included by reference_cpu.cpp (host compiler) AND by
//   kernels.cu / main.cu (nvcc). The GEN_HD macro expands to `__host__
//   __device__` under nvcc and to nothing under the host compiler.
//
// THE REDUCED-SCOPE MODEL (honest teaching simplification)
//   Production de-novo models are deep nets: an RNN/transformer language model
//   over SMILES (REINVENT4) or a diffusion model over 3D graphs (DiffSBDD).
//   Those need cuDNN and days of multi-GPU training. The *teachable* core that
//   fits one deterministic CUDA demo is the SIMPLEST language model that still
//   exhibits the whole pipeline: a FIRST-ORDER MARKOV CHAIN over SMILES
//   characters. We "train" it by counting character->next-character transitions
//   in a tiny corpus (data/sample), then we sample novel strings from it. A
//   transformer replaces this count table with a learned conditional
//   distribution, but the *generation loop is identical* -- "given the context,
//   sample the next token, append, repeat until END". That is the concept this
//   project teaches; THEORY.md §real-world maps each toy piece to its deep-net
//   analogue.
//
// THE GPU PATTERN (PATTERNS.md §1 row "stochastic / Monte-Carlo histories")
//   Each generated molecule is an INDEPENDENT job with its own reproducible RNG
//   stream seeded from (base_seed, molecule_index) -- exactly the per-thread RNG
//   idiom from flagship 5.01 (Monte Carlo dose). One GPU thread generates and
//   scores one molecule; the CPU reference loops the same call. Reproducible
//   per-index seeding is what makes molecule i bit-identical on CPU and GPU.
//
// READ THIS AFTER: util/cuda_check.cuh. Then read kernels.cu and reference_cpu.*.
// ===========================================================================
#pragma once

#include <cstdint>

// Under nvcc, __CUDACC__ is defined and we want these inline functions callable
// from BOTH host and device. Under the plain host compiler the decorators do not
// exist, so the macro expands to nothing. (Idiom from PATTERNS.md §2.)
#ifdef __CUDACC__
#define GEN_HD __host__ __device__
#else
#define GEN_HD
#endif

// ---------------------------------------------------------------------------
// THE SYMBOL ALPHABET
//   A SMILES string is a sequence of these tokens. We use a tiny fixed alphabet
//   so the Markov transition table is small (NSYM x NSYM) and printable. Index 0
//   is a special START/END sentinel ("^"): every string begins in state 0 and we
//   stop when the model emits symbol 0 again (or we hit MAX_LEN). Keeping START
//   and END as the same sentinel makes the model a clean closed loop and matches
//   how language models use a single <bos>/<eos> token class in teaching code.
//
//   The mapping is FIXED and shared, so a symbol's integer id is identical on
//   host and device -- essential for bit-identical generation.
// ---------------------------------------------------------------------------
#define NSYM 14                 // number of symbols including the sentinel
#define SYM_END 0               // id 0 == START and END sentinel ("^")
#define MAX_LEN 64              // hard cap on generated string length (chars)

// The printable character for each symbol id (index == id). Index 0 is the
// sentinel; we render it as '^' only when debugging -- generated molecule
// strings strip it. `static const` (not constexpr) so the same definition is
// safe to include from both the host and device translation units.
GEN_HD inline char sym_char(int id) {
    // A switch keeps this callable on the device without a global array (which
    // would need __constant__/__device__ storage and complicate host use).
    switch (id) {
        case 0:  return '^';   // START/END sentinel (never emitted into output)
        case 1:  return 'C';   // aliphatic carbon
        case 2:  return 'O';   // aliphatic oxygen
        case 3:  return 'N';   // aliphatic nitrogen
        case 4:  return 'c';   // aromatic carbon (lowercase in SMILES)
        case 5:  return 'n';   // aromatic nitrogen
        case 6:  return '(';   // branch open
        case 7:  return ')';   // branch close
        case 8:  return '=';   // double bond
        case 9:  return '1';   // ring-closure digit 1
        case 10: return '2';   // ring-closure digit 2
        case 11: return '3';   // ring-closure digit 3
        case 12: return '4';   // ring-closure digit 4
        case 13: return '#';   // triple bond
        default: return '?';   // unreachable for valid ids
    }
}

// Inverse map: character -> symbol id, or -1 if the character is outside our
// alphabet (such characters are skipped during training). Host-only callers use
// this to parse the training corpus; it is GEN_HD for symmetry but the device
// never needs it.
GEN_HD inline int char_sym(char ch) {
    switch (ch) {
        case 'C': return 1;
        case 'O': return 2;
        case 'N': return 3;
        case 'c': return 4;
        case 'n': return 5;
        case '(': return 6;
        case ')': return 7;
        case '=': return 8;
        case '1': return 9;
        case '2': return 10;
        case '3': return 11;
        case '4': return 12;
        case '#': return 13;
        default:  return -1;   // not in our alphabet -> ignore
    }
}

// ---------------------------------------------------------------------------
// THE RNG: splitmix64, a tiny counter-based generator, IDENTICAL host & device.
//   We need (a) reproducibility and (b) bit-identical streams on CPU and GPU.
//   cuRAND would be the production choice on the device, but it would NOT match
//   a host RNG bit-for-bit, breaking exact verification. splitmix64 is a few
//   integer ops, has good statistical quality for sampling, and runs the same
//   everywhere -- so CPU molecule i == GPU molecule i, exactly. (Same rationale
//   as flagship 5.01.)
// ---------------------------------------------------------------------------
struct Rng { uint64_t state; };

// One splitmix64 step: advance the 64-bit state and return a well-mixed value.
GEN_HD inline uint64_t splitmix64(uint64_t& x) {
    x += 0x9E3779B97F4A7C15ULL;                 // golden-ratio odd increment
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// Seed an INDEPENDENT stream for molecule `index` from a base seed. Mixing the
// index in means molecule 0,1,2,... each get an uncorrelated yet reproducible
// stream -- and the same (base,index) gives the same stream on host and device.
GEN_HD inline Rng rng_seed(uint64_t base, uint64_t index) {
    Rng r;
    r.state = base ^ (index * 0x9E3779B97F4A7C15ULL + 0xD1B54A32D192ED03ULL);
    splitmix64(r.state);       // warm up so nearby seeds diverge immediately
    return r;
}

// A uniform 32-bit random integer (we only need integer comparisons for the
// weighted sampling below, so we avoid floating point entirely -> the sampling
// decision is integer-exact and order-free on host and device).
GEN_HD inline uint32_t rng_u32(Rng& r) {
    return static_cast<uint32_t>(splitmix64(r.state) >> 32);
}

// ---------------------------------------------------------------------------
// THE TRANSITION MODEL  (the "trained generative model")
//   A first-order Markov model is just a table of transition WEIGHTS:
//   weight[from*NSYM + to] = how many times symbol `to` followed symbol `from`
//   in the training corpus (with +1 Laplace smoothing so every transition has
//   non-zero probability and the chain can never get stuck). We also keep the
//   per-row total so sampling is a single integer division-free walk.
//
//   Storing INTEGER counts (not normalised floats) is deliberate: the weighted
//   pick below compares an integer draw against integer cumulative sums, so the
//   sampled token is identical on CPU and GPU with no floating-point rounding to
//   diverge. This is the same "integers commute, floats don't" determinism rule
//   from PATTERNS.md §3.
// ---------------------------------------------------------------------------
struct MarkovModel {
    uint32_t weight[NSYM * NSYM];   // row-major transition counts (+1 smoothed)
    uint32_t row_total[NSYM];       // row_total[s] = sum_t weight[s*NSYM + t]
};

// Sample the next symbol given the current symbol `cur`, using one RNG draw.
//   Classic "roulette wheel" / inverse-CDF sampling over integer weights:
//   draw x in [0, row_total), walk the row accumulating weights, return the
//   token whose cumulative band contains x. O(NSYM) and fully deterministic.
GEN_HD inline int sample_next(const MarkovModel& m, int cur, Rng& rng) {
    const uint32_t total = m.row_total[cur];
    // Reduce a 32-bit draw into [0,total) without floating point. (total is at
    // least NSYM due to Laplace smoothing, so it is never zero.)
    uint32_t x = rng_u32(rng) % total;
    uint32_t acc = 0;
    const uint32_t* row = &m.weight[cur * NSYM];
    for (int t = 0; t < NSYM; ++t) {
        acc += row[t];
        if (x < acc) return t;     // x landed in token t's cumulative band
    }
    return SYM_END;                // numerically unreachable; safe fallback
}

// ---------------------------------------------------------------------------
// GENERATE ONE MOLECULE
//   Autoregressive sampling: start in the sentinel state, repeatedly sample the
//   next symbol and append its character, until we sample the END sentinel or
//   reach MAX_LEN. Writes the ASCII string (NUL-terminated) into out[] and
//   returns its length. This is the exact loop a transformer sampler runs, only
//   with sample_next() standing in for a forward pass + softmax.
//
//   out      : caller-provided buffer of at least MAX_LEN+1 bytes
//   returns  : number of characters written (excluding the NUL terminator)
// ---------------------------------------------------------------------------
GEN_HD inline int generate_molecule(const MarkovModel& m, Rng& rng, char* out) {
    int cur = SYM_END;             // start state == sentinel
    int len = 0;
    for (int step = 0; step < MAX_LEN; ++step) {
        int nxt = sample_next(m, cur, rng);
        if (nxt == SYM_END) break;  // model chose to stop -> molecule complete
        out[len++] = sym_char(nxt);
        cur = nxt;
    }
    out[len] = '\0';
    return len;
}

// ---------------------------------------------------------------------------
// THE SCORING FUNCTION  (the "goal" of goal-directed design)
//   In a real RL fine-tuning loop the reward is a docking score, QED, SA score,
//   etc. Here we use a cheap, deterministic, *integer* drug-likeness proxy so
//   the demo stays self-contained and verifiable. It rewards molecules that look
//   like reasonable small organic structures and penalises obvious nonsense:
//
//     + reward heteroatom content (O, N, n) up to a cap  -> "polarity"
//     + small bonus for a ring-closure digit appearing   -> "has a ring"
//     - penalty if parentheses are unbalanced            -> "invalid branches"
//     - penalty for being too short or too long          -> "drug-like size"
//
//   The score is returned as a fixed-point integer (units: milli-reward) so the
//   top-K reduction is exact and order-independent (PATTERNS.md §3/§4). This is
//   a TOY surrogate, NOT a validated medicinal-chemistry score -- THEORY.md says
//   so plainly and points at QED/SA/docking as the real thing.
//
//   s        : the molecule string (NUL-terminated ASCII)
//   len      : its length (chars), as returned by generate_molecule
//   returns  : an integer reward in milli-units (higher == "more drug-like")
// ---------------------------------------------------------------------------
GEN_HD inline int score_molecule(const char* s, int len) {
    int hetero = 0;       // count of O/N/n (polar) atoms
    int ring_digit = 0;   // saw a ring-closure digit at least once?
    int paren = 0;        // running parenthesis balance
    int paren_bad = 0;    // became negative at any point -> malformed

    for (int i = 0; i < len; ++i) {
        char c = s[i];
        if (c == 'O' || c == 'N' || c == 'n') hetero++;
        else if (c == '1' || c == '2' || c == '3' || c == '4') ring_digit = 1;
        else if (c == '(') paren++;
        else if (c == ')') { paren--; if (paren < 0) paren_bad = 1; }
    }
    if (paren != 0) paren_bad = 1;    // left a branch open at the end

    // Build the integer reward in milli-units. The constants are arbitrary but
    // FIXED, so the ranking is reproducible and the same on CPU and GPU.
    int score = 0;
    int hetero_capped = hetero > 4 ? 4 : hetero;   // diminishing returns past 4
    score += hetero_capped * 250;                  // +0.25 reward per polar atom
    score += ring_digit ? 300 : 0;                 // +0.30 if it has a ring
    if (paren_bad) score -= 1000;                  // -1.0 for malformed branches

    // Drug-like size window: reward 4..40 chars, penalise outside it linearly.
    if (len < 4)       score -= (4 - len) * 200;   // too small
    else if (len > 40) score -= (len - 40) * 50;   // too big
    else               score += 200;               // inside the sweet spot

    return score;
}
