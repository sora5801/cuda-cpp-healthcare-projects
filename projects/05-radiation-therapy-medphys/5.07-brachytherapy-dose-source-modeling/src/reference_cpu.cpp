// ===========================================================================
// src/reference_cpu.cpp  --  The plain-C++ TG-43 baseline we trust
// ---------------------------------------------------------------------------
// Project 5.7 : Brachytherapy Dose & Source Modeling
//
// ROLE IN THE PROJECT
//   This is the "ground truth" the GPU result is checked against. It is written
//   to be OBVIOUSLY correct -- plain nested loops over voxels and dwells, no
//   parallelism, no cleverness -- so that when the GPU and CPU agree, we believe
//   the GPU. The per-voxel dose math is shared with the kernel via
//   tg43_physics.h, so the ONLY differences can be floating-point rounding.
//
//   Compiled by the host C++ compiler only (no CUDA here). See reference_cpu.h.
//
// READ THIS AFTER: reference_cpu.h, tg43_physics.h. Twin: kernels.cu (GPU).
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>     // std::ifstream
#include <sstream>     // std::istringstream (skip comment lines cleanly)
#include <stdexcept>   // std::runtime_error
#include <string>

// ---------------------------------------------------------------------------
// next_token_line: read the next NON-COMMENT, non-blank line from the stream
// into `out`. The sample format (data/README.md) allows '#'-prefixed comment
// lines and blank lines for readability; the parser skips them so the human-
// readable sample and the machine parse stay in sync. Returns false at EOF.
// ---------------------------------------------------------------------------
static bool next_token_line(std::istream& in, std::string& out) {
    std::string line;
    while (std::getline(in, line)) {
        // Trim leading whitespace to test for a comment marker.
        std::size_t p = line.find_first_not_of(" \t\r\n");
        if (p == std::string::npos) continue;   // blank line
        if (line[p] == '#') continue;           // comment line
        out = line;
        return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// require_line: like next_token_line but throws with a helpful message if the
// file ends early -- so a truncated sample fails loudly at load, not later with
// a mysterious wrong number.
// ---------------------------------------------------------------------------
static std::string require_line(std::istream& in, const char* what) {
    std::string s;
    if (!next_token_line(in, s))
        throw std::runtime_error(std::string("plan file truncated: expected ") + what);
    return s;
}

// ---------------------------------------------------------------------------
// load_plan: parse the tiny whitespace text plan (see data/README.md).
//   Layout (comment/blank lines ignored):
//     L Lambda                              # source length[cm], dose-rate const
//     n_g                                   # number of radial-dose samples
//     r_1 g_1  ... (n_g pairs, one per line)
//     n_Fr n_Ft                             # anisotropy grid dims
//     F_r_1 ... F_r_{n_Fr}                  # the anisotropy radii  (one line)
//     F_t_1 ... F_t_{n_Ft}                  # the anisotropy angles (one line)
//     F row for r_1: F_t_1 ... F_t_{n_Ft}   # n_Fr such rows
//     ...
//     nx ny nz  ox oy oz  spacing           # dose grid
//     n_dwells                              # number of dwell positions
//     x y z weight  ... (n_dwells lines)
// ---------------------------------------------------------------------------
Plan load_plan(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open plan file: " + path);

    Plan plan;
    SourceModel& s = plan.source;

    // --- source length + dose-rate constant --------------------------------
    {
        std::istringstream ls(require_line(in, "L Lambda"));
        ls >> s.L >> s.Lambda;
    }

    // --- radial dose function g_L(r) ---------------------------------------
    {
        std::istringstream ls(require_line(in, "n_g"));
        ls >> s.n_g;
        if (s.n_g < 1 || s.n_g > TG43_MAX_RADII)
            throw std::runtime_error("n_g out of range");
        for (int i = 0; i < s.n_g; ++i) {
            std::istringstream rr(require_line(in, "r g"));
            rr >> s.g_r[i] >> s.g_val[i];
        }
    }

    // --- anisotropy grid dimensions + axes ---------------------------------
    {
        std::istringstream ls(require_line(in, "n_Fr n_Ft"));
        ls >> s.n_Fr >> s.n_Ft;
        if (s.n_Fr < 1 || s.n_Fr > TG43_MAX_RADII ||
            s.n_Ft < 1 || s.n_Ft > TG43_MAX_ANGLES)
            throw std::runtime_error("anisotropy dims out of range");
        {
            std::istringstream rr(require_line(in, "F radii"));
            for (int i = 0; i < s.n_Fr; ++i) rr >> s.F_r[i];
        }
        {
            std::istringstream tt(require_line(in, "F angles"));
            for (int j = 0; j < s.n_Ft; ++j) tt >> s.F_t[j];
        }
        // n_Fr rows, each with n_Ft F values (row-major storage).
        for (int i = 0; i < s.n_Fr; ++i) {
            std::istringstream fr(require_line(in, "F row"));
            for (int j = 0; j < s.n_Ft; ++j) fr >> s.F_val[i * s.n_Ft + j];
        }
    }

    // --- dose grid ----------------------------------------------------------
    {
        DoseGrid& g = plan.grid;
        std::istringstream ls(require_line(in, "grid"));
        ls >> g.nx >> g.ny >> g.nz >> g.ox >> g.oy >> g.oz >> g.spacing;
        if (g.nx < 1 || g.ny < 1 || g.nz < 1)
            throw std::runtime_error("grid dims must be positive");
    }

    // --- dwell positions ----------------------------------------------------
    {
        std::istringstream ls(require_line(in, "n_dwells"));
        int nd = 0;
        ls >> nd;
        if (nd < 1 || nd > TG43_MAX_DWELLS)
            throw std::runtime_error("n_dwells out of range");
        plan.dwells.resize(nd);
        for (int k = 0; k < nd; ++k) {
            std::istringstream dd(require_line(in, "dwell x y z weight"));
            dd >> plan.dwells[k].x >> plan.dwells[k].y
               >> plan.dwells[k].z >> plan.dwells[k].weight;
        }
    }

    return plan;
}

// ---------------------------------------------------------------------------
// dose_cpu: the serial reference. Two nested loops:
//   OUTER over every voxel v (the independent work item -> a GPU thread), and
//   INNER over every dwell position (superposition of point/line sources).
// Each inner iteration calls dose_rate_one_dwell() from tg43_physics.h -- the
// EXACT function the kernel calls -- and we accumulate the sum in a `double`
// before storing the result as float. Accumulating in double keeps the CPU and
// GPU sums on equal footing (the kernel does the same), so the tiny residual is
// only FMA/rounding, not an algorithmic gap. See THEORY "How we verify".
//   Complexity: O(n_voxels * n_dwells). The outer loop has no cross-iteration
//   dependencies -- the reason this problem is a textbook GPU fit.
// ---------------------------------------------------------------------------
void dose_cpu(const Plan& plan, std::vector<float>& dose) {
    const DoseGrid& g = plan.grid;
    dose.assign(static_cast<std::size_t>(g.size()), 0.0f);

    for (int iz = 0; iz < g.nz; ++iz) {
        const double pz = g.cz(iz);              // voxel-center world z [cm]
        for (int iy = 0; iy < g.ny; ++iy) {
            const double py = g.cy(iy);          // voxel-center world y [cm]
            for (int ix = 0; ix < g.nx; ++ix) {
                const double px = g.cx(ix);      // voxel-center world x [cm]

                // Superpose the TG-43 dose rate from every dwell position.
                double acc = 0.0;
                for (const Dwell& d : plan.dwells)
                    acc += dose_rate_one_dwell(plan.source, d, px, py, pz);

                const int idx = (iz * g.ny + iy) * g.nx + ix;  // flat index
                dose[static_cast<std::size_t>(idx)] = static_cast<float>(acc);
            }
        }
    }
}
