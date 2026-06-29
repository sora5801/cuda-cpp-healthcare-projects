// ===========================================================================
// src/reference_cpu.cpp  --  Trusted serial enumerator + synthon loader
// ---------------------------------------------------------------------------
// Project 1.18 : Fragment / Combinatorial Library Enumeration
//
// ROLE IN THE PROJECT
//   (1) load_synthons(): parse the tiny text catalog (data/README.md format).
//   (2) enumerate_cpu(): the obviously-correct SERIAL enumeration the GPU kernel
//       is verified against -- one readable loop over all N products, no
//       parallelism, no cleverness. If CPU and GPU agree, we trust the GPU.
//   (3) product_label(): pretty-print a product index for the stdout report.
//
//   The per-product MATH (decode index, sum descriptors, apply filter) is NOT
//   duplicated here -- it is called from product_core.h so the CPU and GPU run
//   byte-identical formulas (PATTERNS.md sec.2). This file only ORCHESTRATES the
//   loop and the reductions.
//
//   Compiled by the host C++ compiler only (no CUDA). See reference_cpu.h.
//
// READ THIS AFTER: product_core.h, reference_cpu.h. Compare against kernels.cu.
// ===========================================================================
#include "reference_cpu.h"

#include <cmath>      // std::llround
#include <fstream>
#include <sstream>
#include <stdexcept>

// ---------------------------------------------------------------------------
// load_synthons: read the synthon catalog. Format (full spec in data/README.md):
//
//   <N_SLOTS>                       # must equal the compiled N_SLOTS (=3)
//   # repeated per slot k = 0..N_SLOTS-1:
//   SLOT <k> <size_k>               # header line announcing this slot
//   <name> <MW> <cLogP> <TPSA> <HBD> <HBA>     # one line per synthon (size_k lines)
//   ...
//
// Lines beginning with '#' and blank lines are comments and skipped. We validate
// N_SLOTS and N_DESC so a mismatched build fails LOUDLY rather than reading
// garbage (CLAUDE.md: reproducibility is sacred).
// ---------------------------------------------------------------------------
SynthonLibrary load_synthons(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open synthon catalog: " + path);

    // Small helper: pull the next NON-comment, non-blank line into `line`.
    // Returns false at end of file. Comments let the sample file be self-
    // documenting without confusing the parser.
    auto next_line = [&](std::string& line) -> bool {
        while (std::getline(in, line)) {
            // Trim a trailing '\r' so files saved on Windows parse on Linux too.
            if (!line.empty() && line.back() == '\r') line.pop_back();
            std::string t = line;
            std::size_t a = t.find_first_not_of(" \t");
            if (a == std::string::npos) continue;          // blank line
            if (t[a] == '#') continue;                     // comment line
            return true;
        }
        return false;
    };

    std::string line;
    if (!next_line(line)) throw std::runtime_error("empty catalog: " + path);

    // First meaningful line: the number of slots. Must match the build.
    int n_slots = 0;
    { std::istringstream ss(line); ss >> n_slots; }
    if (n_slots != N_SLOTS)
        throw std::runtime_error("slot-count mismatch: file has " +
            std::to_string(n_slots) + " but this build expects " +
            std::to_string(N_SLOTS) + " (rebuild with matching N_SLOTS)");

    SynthonLibrary lib;

    // Read each slot's header ("SLOT k size") then its `size` synthon rows.
    for (int k = 0; k < N_SLOTS; ++k) {
        if (!next_line(line))
            throw std::runtime_error("unexpected EOF before slot header " +
                                     std::to_string(k) + " in " + path);
        std::istringstream hs(line);
        std::string tag; int kk = -1, size = -1;
        hs >> tag >> kk >> size;
        if (tag != "SLOT" || kk != k || size <= 0)
            throw std::runtime_error("malformed SLOT header (expected 'SLOT " +
                std::to_string(k) + " <size>') in " + path + " -> '" + line + "'");

        lib.sizes[k] = size;
        lib.desc[k].resize(static_cast<std::size_t>(size) * N_DESC);
        lib.name[k].resize(size);

        // Parse `size` rows of "<name> <MW> <cLogP> <TPSA> <HBD> <HBA>".
        for (int j = 0; j < size; ++j) {
            if (!next_line(line))
                throw std::runtime_error("unexpected EOF reading synthon " +
                    std::to_string(j) + " of slot " + std::to_string(k) + " in " + path);
            std::istringstream rs(line);
            std::string nm;
            double mw, clogp, tpsa, hbd, hba;
            if (!(rs >> nm >> mw >> clogp >> tpsa >> hbd >> hba))
                throw std::runtime_error("malformed synthon row in " + path + " -> '" + line + "'");
            lib.name[k][j] = nm;
            // Store the 5 descriptors in the fixed DescIndex order.
            double* row = &lib.desc[k][static_cast<std::size_t>(j) * N_DESC];
            row[D_MW]    = mw;
            row[D_CLOGP] = clogp;
            row[D_TPSA]  = tpsa;
            row[D_HBD]   = hbd;
            row[D_HBA]   = hba;
        }
    }
    return lib;
}

// ---------------------------------------------------------------------------
// enumerate_cpu: walk EVERY product in flat-index order, decode it into per-slot
// synthon indices, sum the chosen synthons' descriptors, apply the filter, and
// accumulate the deterministic outputs.
//   Complexity: O(N * N_SLOTS * N_DESC) time, O(1) extra space beyond the
//   FIRST_K preview. N = product of slot sizes -- the whole point is that N
//   explodes combinatorially, which is exactly why we want the GPU.
//   Determinism: products are visited in order, the count is integer, and the
//   MW sum is accumulated in FIXED POINT (milli-g/mol) so it is independent of
//   summation order and matches the GPU's integer atomic exactly.
// ---------------------------------------------------------------------------
void enumerate_cpu(const SynthonLibrary& lib, EnumResult& out) {
    out.n_pass = 0;
    out.sum_mw_pass_milli = 0;
    out.first_pass.clear();

    const int64_t N = lib.num_products();
    const int sizes[N_SLOTS] = {lib.sizes[0], lib.sizes[1], lib.sizes[2]};

    for (int64_t p = 0; p < N; ++p) {
        // (1) Decode the flat index into per-slot synthon indices (odometer).
        int idx[N_SLOTS];
        decode_product_index(p, sizes, idx);

        // (2) Gather pointers to the chosen synthons' descriptor rows and sum.
        ProductInputs pin;
        for (int k = 0; k < N_SLOTS; ++k)
            pin.row[k] = &lib.desc[k][static_cast<std::size_t>(idx[k]) * N_DESC];
        double desc[N_DESC];
        accumulate_descriptors(pin, desc);

        // (3) Apply the Lipinski + Veber filter (shared HD function).
        if (passes_filter(desc)) {
            ++out.n_pass;
            // Fixed-point MW accumulation: round MW (g/mol) to milli-g/mol so the
            // running total is an EXACT integer sum (order-independent). llround
            // gives the same nearest-integer the GPU computes from identical input.
            out.sum_mw_pass_milli +=
                static_cast<int64_t>(std::llround(desc[D_MW] * MW_FIXED_SCALE));
            // Record the first FIRST_K passing indices for the preview report.
            if (static_cast<int>(out.first_pass.size()) < FIRST_K)
                out.first_pass.push_back(p);
        }
    }
}

// ---------------------------------------------------------------------------
// product_label: turn a flat product index into "slotA + slotB + slotC" using
// the synthon names, for the human-readable stdout report. Pure host helper.
// ---------------------------------------------------------------------------
std::string product_label(const SynthonLibrary& lib, int64_t p) {
    const int sizes[N_SLOTS] = {lib.sizes[0], lib.sizes[1], lib.sizes[2]};
    int idx[N_SLOTS];
    decode_product_index(p, sizes, idx);
    std::string s;
    for (int k = 0; k < N_SLOTS; ++k) {
        if (k) s += " + ";
        s += lib.name[k][idx[k]];
    }
    return s;
}
