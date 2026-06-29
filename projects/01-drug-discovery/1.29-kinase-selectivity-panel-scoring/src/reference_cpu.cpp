// ===========================================================================
// src/reference_cpu.cpp  --  The trusted serial baseline + dataset loader
// ---------------------------------------------------------------------------
// Project 1.29 : Kinase Selectivity Panel Scoring
//
// ROLE IN THE PROJECT
//   (1) load_panel(): parse the tiny text dataset (data/README.md format) into a
//       KinasePanel (the query compound + N kinase pockets).
//   (2) score_panel_cpu(): the obviously-correct serial computation the GPU
//       kernel is verified against. One readable loop, no parallelism, no
//       cleverness -- when CPU and GPU agree we trust the GPU.
//
//   Both call the SAME __host__ __device__ score_kinase() / predicted_pK_milli()
//   from selectivity_core.h, so the CPU and GPU results are bit-for-bit identical
//   (integer math, no float reordering) -- that is what makes verification EXACT.
//
//   Compiled by the host C++ compiler only (no CUDA). See reference_cpu.h.
//
// READ THIS AFTER: reference_cpu.h, selectivity_core.h. Compare with kernels.cu.
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>     // std::ifstream
#include <sstream>     // std::istringstream (line-by-line parsing)
#include <stdexcept>   // std::runtime_error
#include <string>

// ---------------------------------------------------------------------------
// next_data_line : pull the next NON-COMMENT, non-blank line from the stream into
//   `line`. Lines whose first non-space character is '#' are dataset comments
//   (used in data/sample to label fields) and are skipped. Returns false at EOF.
//   Keeping this tiny helper local makes the loader below read like the file
//   format documented in data/README.md.
// ---------------------------------------------------------------------------
static bool next_data_line(std::istream& in, std::string& line) {
    while (std::getline(in, line)) {
        // Find the first non-whitespace character to test for a comment / blank.
        std::size_t p = line.find_first_not_of(" \t\r\n");
        if (p == std::string::npos) continue;   // blank line -> skip
        if (line[p] == '#') continue;           // comment line -> skip
        return true;
    }
    return false;
}

KinasePanel load_panel(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open kinase panel file: " + path);

    std::string line;

    // ---- header: "N NFEAT" -------------------------------------------------
    if (!next_data_line(in, line))
        throw std::runtime_error("empty panel file (expected '<N> <NFEAT>' header): " + path);
    int n = 0, nfeat = 0;
    {
        std::istringstream hs(line);
        if (!(hs >> n >> nfeat))
            throw std::runtime_error("bad header (expected '<N> <NFEAT>') in " + path);
    }
    if (nfeat != NFEAT)
        throw std::runtime_error("feature-count mismatch: file has " + std::to_string(nfeat) +
                                 " channels but this build expects NFEAT=" + std::to_string(NFEAT) +
                                 " (rebuild with matching NFEAT in selectivity_core.h)");
    if (n <= 0) throw std::runtime_error("non-positive panel size in " + path);

    KinasePanel panel;
    panel.n = n;

    // ---- ligand line: "LIGAND f0 f1 ... f7" --------------------------------
    if (!next_data_line(in, line))
        throw std::runtime_error("missing LIGAND line in " + path);
    {
        std::istringstream ls(line);
        std::string tag;
        ls >> tag;   // the literal word "LIGAND" (a readability tag, not used)
        if (tag != "LIGAND")
            throw std::runtime_error("expected a 'LIGAND ...' line, got '" + tag + "' in " + path);
        for (int f = 0; f < NFEAT; ++f) {
            if (!(ls >> panel.ligand[f]))
                throw std::runtime_error("LIGAND line has fewer than NFEAT feature offers in " + path);
        }
    }

    // ---- N kinase lines: "<name> <bias> r0 r1 ... r7" ----------------------
    panel.pockets.reserve(static_cast<std::size_t>(n));
    panel.names.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        if (!next_data_line(in, line))
            throw std::runtime_error("panel declares " + std::to_string(n) +
                                     " kinases but fewer rows are present in " + path);
        std::istringstream ks(line);
        std::string name;
        KinasePocket pocket{};
        pocket.id = i;                   // stable index used only for ranking/report
        if (!(ks >> name >> pocket.bias))
            throw std::runtime_error("kinase row " + std::to_string(i) +
                                     " is missing its name or bias in " + path);
        for (int f = 0; f < NFEAT; ++f) {
            if (!(ks >> pocket.req[f]))
                throw std::runtime_error("kinase row " + std::to_string(i) +
                                         " has fewer than NFEAT requirements in " + path);
        }
        panel.names.push_back(name);
        panel.pockets.push_back(pocket);
    }
    return panel;
}

// ---------------------------------------------------------------------------
// score_panel_cpu : serial reference. Loop over kinases; for each, run the shared
// score_kinase() physics, map to a predicted pK, and flag hits. Returns the
// integer S-count (number of kinases bound above threshold).
//   Complexity: O(N * NFEAT) time, O(N) output space.
// ---------------------------------------------------------------------------
int32_t score_panel_cpu(const KinasePanel& panel,
                        std::vector<int32_t>& pK_milli,
                        std::vector<int32_t>& hit) {
    pK_milli.assign(static_cast<std::size_t>(panel.n), 0);
    hit.assign(static_cast<std::size_t>(panel.n), 0);

    int32_t s_count = 0;   // numerator of the S-score: how many kinases are "bound"
    for (int i = 0; i < panel.n; ++i) {
        // (1) raw integer match score for this kinase (shared HD physics).
        const int32_t raw = score_kinase(panel.ligand, panel.pockets[static_cast<std::size_t>(i)]);
        // (2) map to a predicted affinity pK in milli-units (exact integer).
        const int32_t pK = predicted_pK_milli(raw);
        // (3) does it clear the selectivity threshold? (pure integer compare).
        const int32_t h = is_hit(pK) ? 1 : 0;
        pK_milli[static_cast<std::size_t>(i)] = pK;
        hit[static_cast<std::size_t>(i)]      = h;
        s_count += h;   // integer accumulation -> deterministic, order-independent
    }
    return s_count;
}
