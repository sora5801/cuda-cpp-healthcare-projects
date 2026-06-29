// ===========================================================================
// src/reference_cpu.h  --  Host data model + CPU reference for de-novo design
// ---------------------------------------------------------------------------
// Project 1.10 : De Novo Generative Molecular Design (reduced-scope teaching).
//
// WHY A SEPARATE HEADER
//   reference_cpu.cpp is compiled by the plain C++ compiler and must NOT see any
//   __global__ syntax, so its prototypes cannot live in kernels.cuh. main.cu and
//   reference_cpu.cpp both include THIS pure-C++ header (plus the shared,
//   host+device generator.h) so they agree on the types and signatures.
//
// THE PIPELINE (shared by CPU and GPU)
//   1. load_corpus()  : parse data/sample into a list of training SMILES + the
//                       run parameters (how many molecules to generate, seed).
//   2. train_model()  : count character transitions -> a MarkovModel (the
//                       "generative model"). Done ONCE on the host; the same
//                       model is then used by the CPU loop and uploaded to the
//                       GPU, so both sample from identical probabilities.
//   3. generate_and_score_cpu() : the trusted baseline -- loop over molecules,
//                       generate+score each with generator.h, fill the outputs.
//   The GPU twin of step 3 lives in kernels.cu; main.cu asserts they agree.
//
// READ THIS BEFORE: reference_cpu.cpp, main.cu, kernels.cuh.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "generator.h"   // MarkovModel, generate_molecule, score_molecule, etc.

// ---------------------------------------------------------------------------
// Corpus: everything parsed from the input file (data/sample/...).
//   train   : the training SMILES strings (used to build the Markov model)
//   n_gen   : how many novel molecules to sample at run time
//   seed    : base RNG seed (reproducible generation)
// ---------------------------------------------------------------------------
struct Corpus {
    std::vector<std::string> train;   // training SMILES strings
    int                      n_gen;   // number of molecules to generate
    uint64_t                 seed;    // base RNG seed
};

// Parse the corpus file. Lines beginning with '#' and blank lines are ignored;
// the first non-comment line is the header "n_train n_generate seed", followed
// by n_train SMILES lines. Throws std::runtime_error on I/O or format errors so
// demos fail loudly. (Defined in reference_cpu.cpp.)
Corpus load_corpus(const std::string& path);

// Build the first-order Markov transition model from the training strings.
//   For each adjacent pair (prev -> next) in "^ S1 S2 ... Sk ^" we increment
//   weight[prev*NSYM + next]; we frame every string with the SYM_END sentinel on
//   both ends so the model learns how molecules START and END. Laplace +1
//   smoothing is applied so no transition is impossible. (Defined in
//   reference_cpu.cpp; pure host code.)
MarkovModel train_model(const Corpus& c);

// CPU reference: generate `n_gen` molecules and score them.
//   model    : the trained transition model
//   n_gen    : number of molecules to generate (== Corpus::n_gen)
//   seed     : base RNG seed; molecule i uses stream rng_seed(seed, i)
//   scores   : OUT, resized to n_gen, integer milli-reward per molecule
//   lengths  : OUT, resized to n_gen, character length of each molecule
//   best_smiles : OUT, the SMILES string of the single highest-scoring molecule
//                 (ties broken by lower index, so it is deterministic)
//   best_index  : OUT, the index of that best molecule
// This is the trusted baseline the GPU result is verified against.
void generate_and_score_cpu(const MarkovModel& model, int n_gen, uint64_t seed,
                            std::vector<int>& scores, std::vector<int>& lengths,
                            std::string& best_smiles, int& best_index);
