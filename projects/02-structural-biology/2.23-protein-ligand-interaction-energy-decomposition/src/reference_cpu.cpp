// ===========================================================================
// src/reference_cpu.cpp  --  The plain-C++ baseline we trust + the data loader
// ---------------------------------------------------------------------------
// Project 2.23 : Protein-Ligand Interaction Energy Decomposition
//
// ROLE IN THE PROJECT
//   (1) load_system(): parse the tiny text trajectory in data/sample into an
//       MmgbsaSystem (format documented in data/README.md).
//   (2) decompose_cpu(): the "ground truth" per-residue MM-GBSA decomposition the
//       GPU result is checked against. It is written to be OBVIOUSLY correct -- a
//       plain triple loop (residue x frame x ligand atom), no parallelism, no
//       cleverness -- so that when the GPU and CPU agree, we believe the GPU.
//
//   The per-pair PHYSICS is NOT duplicated here: both this CPU loop and the GPU
//   kernel call residue_frame_energy() / pair_energy_components() from mmgbsa.h,
//   so they run byte-for-byte the same formula (PATTERNS.md sec 2). That makes
//   the GPU-vs-CPU check a tight tolerance comparison (THEORY.md "How we verify").
//
//   Compiled by the host C++ compiler only (no CUDA here). See reference_cpu.h.
//
// READ THIS AFTER: mmgbsa.h, reference_cpu.h. Compare against kernels.cu (twin).
// ===========================================================================
#include "reference_cpu.h"   // -> mmgbsa.h (data model + HD physics core)

#include <cstddef>     // std::size_t
#include <fstream>     // std::ifstream
#include <sstream>     // std::istringstream
#include <stdexcept>   // std::runtime_error
#include <string>

// ---------------------------------------------------------------------------
// load_system: read the committed text trajectory (data/README.md documents the
// exact grammar). The format is intentionally human-readable so a learner can
// open data/sample and SEE the system. Layout:
//
//   line:  F M L cutoff                     (counts + interaction cutoff in A)
//   M lines: name charge eps rmin_half born      (one per protein residue bead)
//   L lines: charge eps rmin_half born            (one per ligand atom)
//   then F blocks, each block:
//       M lines: x y z                            (residue coords for that frame)
//       L lines: x y z                            (ligand  coords for that frame)
//
// We tolerate blank lines and '#'-comment lines anywhere (skipped) so the sample
// can annotate itself. Throws std::runtime_error with a precise message on any
// malformed input so demos fail loudly rather than silently on garbage.
// ---------------------------------------------------------------------------

// next_data_line: pull the next non-blank, non-comment line from the stream.
// Returns false at end of file. Centralising this keeps the parser readable and
// lets the sample file carry '#' comments that explain each section.
static bool next_data_line(std::istream& in, std::string& line) {
    while (std::getline(in, line)) {
        // Strip a trailing CR so files saved on Windows (CRLF) parse on Linux.
        if (!line.empty() && line.back() == '\r') line.pop_back();
        // Find the first non-space char; skip blank lines and '#' comments.
        std::size_t i = line.find_first_not_of(" \t");
        if (i == std::string::npos) continue;      // all whitespace
        if (line[i] == '#') continue;              // comment line
        return true;
    }
    return false;
}

MmgbsaSystem load_system(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open system file: " + path);

    std::string line;
    // --- header: F M L cutoff ------------------------------------------
    if (!next_data_line(in, line))
        throw std::runtime_error("empty system file: " + path);
    MmgbsaSystem sys;
    {
        std::istringstream ss(line);
        if (!(ss >> sys.F >> sys.M >> sys.L >> sys.cutoff))
            throw std::runtime_error("bad header (expected 'F M L cutoff') in " + path);
    }
    if (sys.F <= 0 || sys.M <= 0 || sys.L <= 0)
        throw std::runtime_error("non-positive F/M/L in header of " + path);
    if (sys.cutoff <= 0.0)
        throw std::runtime_error("non-positive cutoff in header of " + path);

    // --- M residue parameter rows --------------------------------------
    sys.res.resize(static_cast<std::size_t>(sys.M));
    for (int m = 0; m < sys.M; ++m) {
        if (!next_data_line(in, line))
            throw std::runtime_error("missing residue parameter rows in " + path);
        std::istringstream ss(line);
        std::string name;
        ResidueParams rp{};
        if (!(ss >> name >> rp.charge >> rp.eps >> rp.rmin_half >> rp.born))
            throw std::runtime_error("bad residue param row " + std::to_string(m) + " in " + path);
        // Copy the (display-only) name into the fixed-size buffer, NUL-terminated.
        // We copy by hand (not strncpy) to dodge MSVC's C4996 deprecation, which
        // this build promotes to an error, and to keep the truncation explicit.
        const std::size_t cap = sizeof(rp.name) - 1;   // leave room for the NUL
        std::size_t k = 0;
        for (; k < cap && k < name.size(); ++k) rp.name[k] = name[k];
        rp.name[k] = '\0';
        sys.res[static_cast<std::size_t>(m)] = rp;
    }

    // --- L ligand parameter rows ---------------------------------------
    sys.lig.resize(static_cast<std::size_t>(sys.L));
    for (int a = 0; a < sys.L; ++a) {
        if (!next_data_line(in, line))
            throw std::runtime_error("missing ligand parameter rows in " + path);
        std::istringstream ss(line);
        LigandParams lp{};
        if (!(ss >> lp.charge >> lp.eps >> lp.rmin_half >> lp.born))
            throw std::runtime_error("bad ligand param row " + std::to_string(a) + " in " + path);
        sys.lig[static_cast<std::size_t>(a)] = lp;
    }

    // --- F frames of coordinates ---------------------------------------
    // res_xyz is [F*M*3], lig_xyz is [F*L*3], both flat row-major (mmgbsa.h).
    sys.res_xyz.resize(static_cast<std::size_t>(sys.F) * sys.M * 3);
    sys.lig_xyz.resize(static_cast<std::size_t>(sys.F) * sys.L * 3);
    auto read_xyz = [&](double* dst, int count, const char* what, int frame) {
        for (int i = 0; i < count; ++i) {
            if (!next_data_line(in, line))
                throw std::runtime_error(std::string("missing ") + what + " coords in frame "
                                         + std::to_string(frame) + " of " + path);
            std::istringstream ss(line);
            if (!(ss >> dst[i * 3 + 0] >> dst[i * 3 + 1] >> dst[i * 3 + 2]))
                throw std::runtime_error(std::string("bad ") + what + " coord line in frame "
                                         + std::to_string(frame) + " of " + path);
        }
    };
    for (int f = 0; f < sys.F; ++f) {
        read_xyz(&sys.res_xyz[static_cast<std::size_t>(f) * sys.M * 3], sys.M, "residue", f);
        read_xyz(&sys.lig_xyz[static_cast<std::size_t>(f) * sys.L * 3], sys.L, "ligand",  f);
    }
    return sys;
}

// ---------------------------------------------------------------------------
// decompose_cpu: the trusted serial decomposition.
//   For each residue m: accumulate its energy components over ALL frames, then
//   divide by F to get the trajectory AVERAGE (the conventional MM-GBSA estimate
//   <E> over an ensemble of snapshots). out[m] holds the four numbers
//   (elec, vdw, gb, total) for residue m.
//
//   Complexity: O(M * F * L) pair evaluations -- exactly the GPU's total work.
//   The OUTER loop is over residues so this mirrors the GPU's "one thread per
//   residue" mapping (kernels.cu): each residue is an independent accumulation,
//   which is precisely why the problem parallelises.
//
//   Determinism: the frame and ligand loops run in fixed index order, and we
//   accumulate in double precision, so the result is reproducible run to run and
//   matches the GPU to within FMA-level rounding (THEORY.md "How we verify").
// ---------------------------------------------------------------------------
void decompose_cpu(const MmgbsaSystem& sys, std::vector<PerResidueEnergy>& out) {
    out.assign(static_cast<std::size_t>(sys.M), PerResidueEnergy{});
    const double cutoff2 = sys.cutoff * sys.cutoff;   // compare squared distances
    const double inv_F   = 1.0 / static_cast<double>(sys.F);  // averaging factor

    for (int m = 0; m < sys.M; ++m) {
        double elec = 0.0, vdw = 0.0, gb = 0.0;   // sums over frames for residue m
        for (int f = 0; f < sys.F; ++f) {
            // Pointers to THIS frame's residue/ligand coordinate blocks.
            const double* res_f = &sys.res_xyz[static_cast<std::size_t>(f) * sys.M * 3];
            const double* lig_f = &sys.lig_xyz[static_cast<std::size_t>(f) * sys.L * 3];
            // The shared HD core (same call the GPU thread makes). It ADDS into
            // elec/vdw/gb, so across frames they accumulate the frame sum.
            residue_frame_energy(sys.res.data(), sys.lig.data(), res_f, lig_f,
                                 m, sys.L, cutoff2, elec, vdw, gb);
        }
        // Average over frames -> the per-residue contribution.
        PerResidueEnergy e;
        e.elec  = elec * inv_F;
        e.vdw   = vdw  * inv_F;
        e.gb    = gb   * inv_F;
        e.total = e.elec + e.vdw + e.gb;
        out[static_cast<std::size_t>(m)] = e;
    }
}
