// ===========================================================================
// src/reference_cpu.cpp  --  Trusted serial shape-overlap baseline + loader
// ---------------------------------------------------------------------------
// Project 1.13 : Pharmacophore & 3D Shape Screening
//
// ROLE
//   (1) load_conformers(): parse the tiny text dataset (data/README.md format)
//       into a ConformerSet, converting each atom's van der Waals radius to a
//       Gaussian alpha via atom_alpha() (shape_overlap.h).
//   (2) shape_tanimoto_cpu(): the obviously-correct serial computation the GPU
//       kernel is verified against. It calls the SAME shared physics functions
//       (molecule_overlap / shape_tanimoto) the GPU kernel calls, so if the two
//       agree we trust the GPU (PATTERNS.md sec 2 + sec 4).
//
//   Compiled by the host C++ compiler only (no CUDA). See reference_cpu.h.
//
// READ THIS AFTER: reference_cpu.h, shape_overlap.h. Compare against kernels.cu.
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>     // std::ifstream
#include <sstream>     // std::istringstream
#include <stdexcept>   // std::runtime_error
#include <string>

// ---------------------------------------------------------------------------
// read_data_line: return the next "real" line (skipping blanks and '#' comment
//   lines) as an istringstream the caller can pull fields from. Centralizing
//   the comment-skipping here keeps the molecule parser below clean.
//   Returns false at end of file.
// ---------------------------------------------------------------------------
static bool read_data_line(std::ifstream& in, std::istringstream& fields) {
    std::string line;
    while (std::getline(in, line)) {
        // Strip a trailing CR so Windows-edited files parse on any platform.
        if (!line.empty() && line.back() == '\r') line.pop_back();
        // Find the first non-space character to detect blank/comment lines.
        std::size_t first = line.find_first_not_of(" \t");
        if (first == std::string::npos) continue;     // blank line -> skip
        if (line[first] == '#') continue;             // comment line -> skip
        fields.clear();
        fields.str(line);
        return true;
    }
    return false;   // end of file
}

// ---------------------------------------------------------------------------
// read_molecule: parse one molecule block (count, label, then that many atom
// lines) into `mol`, returning its label via `label`. Converts each atom's
// radius to a Gaussian alpha here so the rest of the program is radius-free.
// ---------------------------------------------------------------------------
static void read_molecule(std::ifstream& in, Molecule& mol, std::string& label,
                          const std::string& path) {
    std::istringstream fields;

    // (a) atom count M
    if (!read_data_line(in, fields)) throw std::runtime_error("unexpected EOF (atom count) in " + path);
    int m = 0;
    if (!(fields >> m)) throw std::runtime_error("bad atom count in " + path);
    if (m <= 0)         throw std::runtime_error("non-positive atom count in " + path);
    if (m > MAX_ATOMS)  throw std::runtime_error("molecule exceeds MAX_ATOMS (" +
                                                 std::to_string(MAX_ATOMS) + ") in " + path);
    mol.n_atoms = m;

    // (b) label (one whitespace-free token)
    if (!read_data_line(in, fields)) throw std::runtime_error("unexpected EOF (label) in " + path);
    if (!(fields >> label))          throw std::runtime_error("missing molecule label in " + path);

    // (c) M atom lines: x y z radius. We store alpha = atom_alpha(radius).
    for (int a = 0; a < m; ++a) {
        if (!read_data_line(in, fields)) throw std::runtime_error("unexpected EOF (atom row) in " + path);
        double x, y, z, r;
        if (!(fields >> x >> y >> z >> r))
            throw std::runtime_error("bad atom row (expected 'x y z radius') in " + path);
        if (r <= 0.0) throw std::runtime_error("non-positive atom radius in " + path);
        mol.atom[a].x     = x;
        mol.atom[a].y     = y;
        mol.atom[a].z     = z;
        mol.atom[a].alpha = atom_alpha(r);   // radius -> Gaussian width, done once
    }
}

ConformerSet load_conformers(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open conformer file: " + path);

    std::istringstream fields;

    // (1) N = number of library conformers.
    if (!read_data_line(in, fields)) throw std::runtime_error("empty file: " + path);
    int n = 0;
    if (!(fields >> n)) throw std::runtime_error("bad header (expected '<N>') in " + path);
    if (n <= 0)         throw std::runtime_error("non-positive conformer count in " + path);

    ConformerSet set;
    set.n = n;

    // (2) the query molecule (its label is read but not otherwise needed).
    std::string query_label;
    read_molecule(in, set.query, query_label, path);

    // (3) the N library conformers.
    set.lib.resize(static_cast<std::size_t>(n));
    set.name.resize(static_cast<std::size_t>(n));
    for (int k = 0; k < n; ++k) {
        read_molecule(in, set.lib[static_cast<std::size_t>(k)],
                      set.name[static_cast<std::size_t>(k)], path);
    }
    return set;
}

// ---------------------------------------------------------------------------
// shape_tanimoto_cpu: loop the shared physics over every conformer.
//   This is intentionally plain -- no SIMD, no threads. Its job is to be
//   OBVIOUSLY correct, not fast, so it can serve as the ground truth.
// ---------------------------------------------------------------------------
void shape_tanimoto_cpu(const ConformerSet& set, std::vector<double>& out) {
    out.assign(static_cast<std::size_t>(set.n), 0.0);

    // The query self-overlap O_AA is the SAME for every library conformer, so
    // compute it exactly once (the GPU wrapper does the identical hoist).
    const double o_aa = molecule_overlap(set.query, set.query);

    for (int k = 0; k < set.n; ++k) {
        const Molecule& B = set.lib[static_cast<std::size_t>(k)];
        const double o_ab = molecule_overlap(set.query, B);   // cross overlap
        const double o_bb = molecule_overlap(B, B);           // fit self-overlap
        // shape_tanimoto() is the SAME inline used by the GPU kernel -> identical math.
        out[static_cast<std::size_t>(k)] = shape_tanimoto(o_ab, o_aa, o_bb);
    }
}
