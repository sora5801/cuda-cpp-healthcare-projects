// ===========================================================================
// src/reference_cpu.cpp  --  Plain-C++ reference: load, train, generate, score
// ---------------------------------------------------------------------------
// Project 1.10 : De Novo Generative Molecular Design (reduced-scope teaching).
//
// This file is compiled by the HOST C++ compiler (cl.exe / g++), never nvcc, so
// it contains zero CUDA syntax. It implements the trusted baseline that the GPU
// kernel is checked against, plus the shared corpus-loading and model-training
// steps (which run once on the host for both paths). Every molecule is generated
// with the SAME shared functions from generator.h that the GPU kernel uses, so
// CPU molecule i and GPU molecule i are bit-identical.
//
// Read generator.h first (it defines the model + RNG + scorer), then this file,
// then kernels.cu to see the GPU twin.
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>     // std::ifstream
#include <sstream>     // std::istringstream
#include <stdexcept>   // std::runtime_error
#include <climits>     // INT32_MIN
#include <cstdint>     // uint64_t

// ---------------------------------------------------------------------------
// load_corpus: parse the input file into a Corpus.
//   File format (see scripts/make_synthetic.py): optional '#' comment lines,
//   then a header line "n_train n_generate seed", then n_train SMILES lines.
//   We skip blanks/comments so the human-readable header in the sample file is
//   ignored gracefully.
// ---------------------------------------------------------------------------
Corpus load_corpus(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open corpus file: " + path);

    Corpus c;
    std::string line;
    bool have_header = false;
    int  n_train = 0;
    int  read_train = 0;

    while (std::getline(in, line)) {
        // Trim a trailing '\r' so files authored on Windows parse on Linux too.
        if (!line.empty() && line.back() == '\r') line.pop_back();
        // Skip blank lines and '#' comments.
        if (line.empty()) continue;
        if (line[0] == '#') continue;

        if (!have_header) {
            // First real line is the header: three integers.
            std::istringstream iss(line);
            unsigned long long seed_tmp = 0;
            if (!(iss >> n_train >> c.n_gen >> seed_tmp))
                throw std::runtime_error("malformed header (need 'n_train n_generate seed'): " + line);
            c.seed = static_cast<uint64_t>(seed_tmp);
            if (n_train <= 0 || c.n_gen <= 0)
                throw std::runtime_error("header counts must be positive");
            c.train.reserve(static_cast<size_t>(n_train));
            have_header = true;
            continue;
        }

        // Subsequent non-comment lines are training SMILES.
        if (read_train < n_train) {
            c.train.push_back(line);
            ++read_train;
        }
    }

    if (!have_header) throw std::runtime_error("no header line found in corpus");
    if (read_train != n_train)
        throw std::runtime_error("corpus declared " + std::to_string(n_train) +
                                 " training strings but found " + std::to_string(read_train));
    return c;
}

// ---------------------------------------------------------------------------
// train_model: count first-order character transitions into a MarkovModel.
//   For each training string S we walk the sentinel-framed sequence
//   "^ s0 s1 ... s_{k-1} ^" and increment weight[prev -> next] for every
//   adjacent pair. Framing with SYM_END on both ends teaches the model which
//   symbols typically START a molecule (transitions out of the sentinel) and
//   which END it (transitions into the sentinel). Characters outside our
//   alphabet (e.g. brackets we did not model) are skipped.
//
//   Laplace +1 smoothing: every cell starts at 1 so no transition has zero
//   probability -- this guarantees row_total[s] >= NSYM > 0, so sample_next()
//   never divides by zero and the chain can always proceed.
// ---------------------------------------------------------------------------
MarkovModel train_model(const Corpus& c) {
    MarkovModel m;

    // Laplace smoothing: initialise every transition count to 1.
    for (int i = 0; i < NSYM * NSYM; ++i) m.weight[i] = 1u;

    // Accumulate observed transitions from the training corpus.
    for (const std::string& s : c.train) {
        int prev = SYM_END;                  // every string starts at the sentinel
        for (char ch : s) {
            int sym = char_sym(ch);          // map character -> symbol id
            if (sym < 0) continue;           // not in our alphabet: skip it
            m.weight[prev * NSYM + sym] += 1u;
            prev = sym;
        }
        m.weight[prev * NSYM + SYM_END] += 1u;  // and ends at the sentinel
    }

    // Precompute per-row totals so sampling is a single integer walk.
    for (int s = 0; s < NSYM; ++s) {
        uint32_t total = 0;
        for (int t = 0; t < NSYM; ++t) total += m.weight[s * NSYM + t];
        m.row_total[s] = total;
    }
    return m;
}

// ---------------------------------------------------------------------------
// generate_and_score_cpu: the trusted baseline.
//   Loop over the requested number of molecules; each one gets its own
//   reproducible RNG stream seeded from (seed, i). Generate the string with the
//   shared generator.h loop, score it, record score+length, and track the single
//   best (highest score, ties -> lower index) so we can print a concrete example
//   of "goal-directed" output. The GPU kernel performs the IDENTICAL work in
//   parallel; main.cu asserts the score/length arrays match exactly.
// ---------------------------------------------------------------------------
void generate_and_score_cpu(const MarkovModel& model, int n_gen, uint64_t seed,
                            std::vector<int>& scores, std::vector<int>& lengths,
                            std::string& best_smiles, int& best_index) {
    scores.assign(static_cast<size_t>(n_gen), 0);
    lengths.assign(static_cast<size_t>(n_gen), 0);

    char buf[MAX_LEN + 1];        // scratch buffer for one molecule's characters
    int  best_score = INT_MIN;    // sentinel so the first molecule always wins
    best_index = 0;
    best_smiles.clear();

    for (int i = 0; i < n_gen; ++i) {
        Rng rng = rng_seed(seed, static_cast<uint64_t>(i));  // per-molecule stream
        int len = generate_molecule(model, rng, buf);        // sample the string
        int sc  = score_molecule(buf, len);                  // evaluate the reward
        scores[static_cast<size_t>(i)]  = sc;
        lengths[static_cast<size_t>(i)] = len;
        // Track the best molecule deterministically (strictly-greater keeps the
        // lowest index on ties, matching the GPU reduction in main.cu).
        if (sc > best_score) {
            best_score = sc;
            best_index = i;
            best_smiles.assign(buf, buf + len);
        }
    }
}
