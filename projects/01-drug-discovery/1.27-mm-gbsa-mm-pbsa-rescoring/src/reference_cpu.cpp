// ===========================================================================
// src/reference_cpu.cpp  --  The plain-C++ baseline we trust + the data loader
// ---------------------------------------------------------------------------
// Project 1.27 : MM-GBSA / MM-PBSA Rescoring
//
// ROLE IN THE PROJECT
//   (1) load_complex(): parse the tiny text dataset (format in data/README.md)
//       into the Complex struct (receptor atoms + per-snapshot ligand atoms).
//   (2) rescore_cpu(): the "ground truth" the GPU result is checked against. It
//       is OBVIOUSLY correct -- a single readable loop over snapshots that calls
//       the SHARED snapshot_dg() physics from reference_cpu.h -- so when the GPU
//       and CPU agree, we believe the GPU.
//   (3) mean(): the ensemble average that turns per-snapshot dG into the single
//       MM-GBSA binding free-energy estimate.
//
//   Compiled by the host C++ compiler only (no CUDA here). Because it calls the
//   SAME snapshot_dg() that the kernel calls, the CPU and GPU do bit-near
//   identical arithmetic -- the whole point of the shared __host__ __device__
//   header (docs/PATTERNS.md §2).
//
// READ THIS AFTER: reference_cpu.h. Compare against kernels.cu (the GPU twin).
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>     // std::ifstream
#include <sstream>     // std::istringstream (per-line token parsing)
#include <stdexcept>   // std::runtime_error
#include <string>

// ---------------------------------------------------------------------------
// read_atom : parse one whitespace-separated record of 8 doubles into an Atom.
//   The dataset stores each atom as:  x y z  q  sigma eps  born
//   We read field by field so a malformed line fails loudly (with the file path)
//   instead of silently producing garbage energies.
// ---------------------------------------------------------------------------
static Atom read_atom(std::istream& in, const std::string& path) {
    Atom a{};
    if (!(in >> a.x >> a.y >> a.z >> a.q >> a.sigma >> a.eps >> a.born))
        throw std::runtime_error("malformed/short atom record in " + path);
    return a;
}

// ---------------------------------------------------------------------------
// load_complex : read the whole problem from the text format in data/README.md:
//   line 1 (header):  R  L  S  minus_TdS
//   next R lines   :  receptor atoms (8 fields each)
//   next S*L lines :  ligand atoms, grouped by snapshot (snapshot 0's L atoms,
//                     then snapshot 1's L atoms, ...). Row-major into
//                     ligand_snapshots, which is exactly the GPU upload layout.
// Blank lines and '#'-comment lines are skipped so the sample file can be
// annotated for the learner. Throws on any structural problem.
// ---------------------------------------------------------------------------
Complex load_complex(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open complex file: " + path);

    // Helper: pull the next non-blank, non-comment line into an istringstream so
    // we can tokenize it. Returns false at end-of-file.
    auto next_line = [&](std::istringstream& ss) -> bool {
        std::string line;
        while (std::getline(in, line)) {
            // Trim a leading '#': anything after it on the line is a comment.
            std::string::size_type hash = line.find('#');
            if (hash != std::string::npos) line.erase(hash);
            // Skip lines that are now empty / all whitespace.
            if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
            ss.clear();
            ss.str(line);
            return true;
        }
        return false;
    };

    // --- header ------------------------------------------------------------
    Complex cx;
    {
        std::istringstream ss;
        if (!next_line(ss)) throw std::runtime_error("empty file: " + path);
        if (!(ss >> cx.R >> cx.L >> cx.S >> cx.minus_TdS))
            throw std::runtime_error("bad header (expected 'R L S minus_TdS') in " + path);
        if (cx.R <= 0 || cx.L <= 0 || cx.S <= 0)
            throw std::runtime_error("non-positive R/L/S in " + path);
    }

    // --- receptor atoms ----------------------------------------------------
    cx.receptor.reserve(static_cast<std::size_t>(cx.R));
    for (int j = 0; j < cx.R; ++j) {
        std::istringstream ss;
        if (!next_line(ss)) throw std::runtime_error("missing receptor atoms in " + path);
        cx.receptor.push_back(read_atom(ss, path));
    }

    // --- ligand snapshots (S * L atoms, row-major by snapshot) -------------
    cx.ligand_snapshots.reserve(static_cast<std::size_t>(cx.S) * cx.L);
    for (long k = 0; k < static_cast<long>(cx.S) * cx.L; ++k) {
        std::istringstream ss;
        if (!next_line(ss)) throw std::runtime_error("missing ligand-snapshot atoms in " + path);
        cx.ligand_snapshots.push_back(read_atom(ss, path));
    }
    return cx;
}

// ---------------------------------------------------------------------------
// rescore_cpu : the serial reference. For each snapshot s, call the SHARED
// snapshot_dg() (defined in reference_cpu.h) on that snapshot's slice of the
// ligand array. The result dg[s] is the per-frame binding-energy estimate.
//   Complexity: O(S * R * L). This nested-loop baseline is the timing reference
//   that makes the GPU speed-up legible, and the correctness oracle for verify.
// ---------------------------------------------------------------------------
void rescore_cpu(const Complex& cx, std::vector<double>& dg) {
    dg.assign(static_cast<std::size_t>(cx.S), 0.0);
    for (int s = 0; s < cx.S; ++s) {
        // Base pointer to snapshot s's L ligand atoms inside the flat array.
        // Index arithmetic mirrors EXACTLY what the kernel does on the device, so
        // the same atoms are summed in the same order on both sides.
        const Atom* ligand = cx.ligand_snapshots.data()
                           + static_cast<std::size_t>(s) * cx.L;
        dg[static_cast<std::size_t>(s)] =
            snapshot_dg(cx.receptor.data(), cx.R, ligand, cx.L, cx.minus_TdS);
    }
}

// ---------------------------------------------------------------------------
// mean : average the per-snapshot dG values in a FIXED (index 0..n-1) order.
//   We accumulate in a double and divide once at the end. A fixed accumulation
//   order means the host's ensemble average and any host-side recomputation from
//   the GPU's per-snapshot results match; it is also how MMPBSA.py reports the
//   final dG_bind (the trajectory average). Returns 0 for an empty vector.
// ---------------------------------------------------------------------------
double mean(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    double acc = 0.0;
    for (double x : v) acc += x;     // fixed left-to-right summation
    return acc / static_cast<double>(v.size());
}
