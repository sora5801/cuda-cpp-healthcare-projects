// ===========================================================================
// src/reference_cpu.cpp  --  Trusted serial screening baseline + data loader
// ---------------------------------------------------------------------------
// Project 1.4 : Ultra-Large Virtual Screening
//
// ROLE
//   (1) load_library(): parse the tiny text dataset (format in data/README.md).
//   (2) screen_cpu():  the obviously-correct serial pass the GPU is verified
//       against. It just loops over ligands calling score_ligand() -- the SAME
//       __host__ __device__ function the kernel calls -- so if CPU and GPU agree
//       we trust the GPU. No cleverness on purpose.
//
//   Compiled by the host C++ compiler only (no CUDA). See reference_cpu.h.
//
// READ THIS AFTER: screen_core.h, reference_cpu.h. Compare to kernels.cu (twin).
// ===========================================================================
#include "reference_cpu.h"

#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
// next_data_line: read the next NON-COMMENT, non-blank line into `out`.
//   The sample file uses '#'-prefixed comment lines to document its columns;
//   those (and blank lines) must be skipped so the loader sees only data rows.
//   Returns false at end-of-file. Centralising this keeps the parser readable.
// ---------------------------------------------------------------------------
static bool next_data_line(std::istream& in, std::string& out) {
    std::string line;
    while (std::getline(in, line)) {
        // Find the first non-whitespace character to classify the line.
        std::size_t p = line.find_first_not_of(" \t\r\n");
        if (p == std::string::npos) continue;   // blank line -> skip
        if (line[p] == '#') continue;            // comment line -> skip
        out = line;
        return true;
    }
    return false;   // end of file
}

LigandLibrary load_library(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open ligand library file: " + path);

    std::string line;

    // ---- record 1: the ligand count `n` -----------------------------------
    if (!next_data_line(in, line))
        throw std::runtime_error("empty/headerless library file: " + path);
    int n = 0;
    {
        std::istringstream ss(line);
        if (!(ss >> n) || n <= 0)
            throw std::runtime_error("bad ligand count (expected a positive int) in " + path);
    }

    // ---- record 2: the TARGET row -----------------------------------------
    // Layout: "TARGET <mw_opt> <logp_opt_x100> <psa_opt> <feat_required_hex>"
    LigandLibrary lib;
    if (!next_data_line(in, line))
        throw std::runtime_error("missing TARGET line in " + path);
    {
        std::istringstream ss(line);
        std::string tag, feat_hex;
        if (!(ss >> tag >> lib.target.mw_opt >> lib.target.logp_opt_x100
                 >> lib.target.psa_opt >> feat_hex) || tag != "TARGET")
            throw std::runtime_error("malformed TARGET line in " + path);
        // The feature bitmask is written as hex (e.g. "0x1A3F"); std::stoul base 16.
        lib.target.feat_required =
            static_cast<uint32_t>(std::stoul(feat_hex, nullptr, 16));
    }

    // ---- records 3..n+2: one ligand each ----------------------------------
    // Layout: "<mw> <logp_x100> <hbd> <hba> <rotb> <psa> <feat_hex>"
    lib.ligands.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        if (!next_data_line(in, line))
            throw std::runtime_error("file ended before reading all " +
                                     std::to_string(n) + " ligands: " + path);
        std::istringstream ss(line);
        Ligand L{};
        std::string feat_hex;
        if (!(ss >> L.mw >> L.logp_x100 >> L.hbd >> L.hba >> L.rotb >> L.psa >> feat_hex))
            throw std::runtime_error("malformed ligand row " + std::to_string(i) +
                                     " in " + path);
        L.feat = static_cast<uint32_t>(std::stoul(feat_hex, nullptr, 16));
        lib.ligands.push_back(L);
    }
    return lib;
}

void screen_cpu(const LigandLibrary& lib, std::vector<int>& score) {
    score.assign(static_cast<std::size_t>(lib.n()), 0);
    // The whole reference is this loop: score each ligand independently with the
    // SHARED score_ligand() (filter cascade + surrogate dock). Because that
    // function is integer-only, every value here is reproducible and will match
    // the GPU exactly -- which is the entire point of the __host__ __device__
    // core (screen_core.h). The kernel (kernels.cu) is this loop, one iteration
    // per thread.
    for (int i = 0; i < lib.n(); ++i) {
        score[static_cast<std::size_t>(i)] = score_ligand(lib.ligands[i], lib.target);
    }
}
