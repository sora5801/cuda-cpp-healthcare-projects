// ===========================================================================
// src/reference_cpu.h  --  Data model + CPU reference for TransE link prediction
// ---------------------------------------------------------------------------
// Project 1.19 : Network / Polypharmacology Modeling
//
// WHY A PURE-C++ HEADER
//   reference_cpu.cpp is compiled by the host C++ compiler and must not see any
//   CUDA syntax, so the shared DATA MODEL (the KnowledgeGraph container, the file
//   loader) and the CPU reference prototype live here. The GPU side (kernels.cuh)
//   also includes this header to reuse the KnowledgeGraph type -- nothing
//   CUDA-specific leaks in either direction. The actual per-candidate math is
//   shared separately in transe.h (the __host__ __device__ core), so the CPU and
//   GPU compute identical numbers.
//
// THE PROBLEM (see ../THEORY.md for the full derivation)
//   Polypharmacology asks: given a drug, which proteins does it (also) bind?
//   We model the drug-target world as a KNOWLEDGE GRAPH whose entities (drugs,
//   proteins) and relations (e.g. TARGETS) are embedded as d-dim vectors by
//   TransE, which trains them so that  head + relation ~= tail  for true facts.
//   LINK PREDICTION then scores a query drug under the TARGETS relation against
//   every protein tail and ranks them: the top tails are predicted (off-)targets.
//   Each candidate is INDEPENDENT -> ideal data parallelism (one thread/tail).
//
//   This project ships PRE-TRAINED, SYNTHETIC embeddings and performs the SCORING
//   + RANKING on the GPU (training is the research-grade part, described in
//   THEORY.md "Where this sits in the real world"). The synthetic embeddings have
//   a known answer baked in (data/README.md), so the demo is self-checking.
//
// READ THIS BEFORE: reference_cpu.cpp, transe.h, kernels.cuh.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// KnowledgeGraph: one loaded link-prediction query.
//   * head     : the query drug embedding         (length `dim`)
//   * relation : the TARGETS relation embedding    (length `dim`)
//   * tails    : n candidate protein embeddings, ROW-MAJOR and FLATTENED, so tail
//                j occupies tails[j*dim .. j*dim + dim - 1]. Row-major + flat is
//                the layout the GPU wants (one contiguous device array) AND the
//                simplest thing for the CPU loop -- one shape, no copies.
//   * true_targets : the ground-truth target indices baked into the synthetic
//                data (used only to REPORT recovery; never fed to the scorer).
// ---------------------------------------------------------------------------
struct KnowledgeGraph {
    int n   = 0;                       // number of candidate protein tails
    int dim = 0;                       // embedding dimension d
    std::vector<float> head;           // [dim]     query drug embedding h
    std::vector<float> relation;       // [dim]     TARGETS relation embedding r
    std::vector<float> tails;          // [n * dim] candidate tails, row-major
    std::vector<int>   true_targets;   // ground-truth target indices (for reporting)
};

// Load a KnowledgeGraph from the text format documented in data/README.md:
//   line 1:  "<n> <dim>"
//   line 2:  the head (drug) embedding       : dim floats
//   line 3:  the relation (TARGETS) embedding : dim floats
//   line 4:  "<n_true> <idx0> <idx1> ..."    : ground-truth target indices
//   next n:  each candidate tail embedding   : dim floats
// Throws std::runtime_error on a missing/ill-formed file.
KnowledgeGraph load_knowledge_graph(const std::string& path);

// CPU reference: fill score[j] with the TransE plausibility score of candidate
// tail j (the negative squared distance from transe.h). This is the trusted,
// obviously-correct baseline the GPU result is checked against, and the timing
// baseline that makes the speed-up legible. `score` is resized to kg.n.
void transe_score_cpu(const KnowledgeGraph& kg, std::vector<float>& score);
