// ===========================================================================
// src/reference_cpu.cpp  --  Plan loader + serial pencil-beam dose reference
// ---------------------------------------------------------------------------
// Project 5.3 : Proton & Heavy-Ion Therapy Dose
//
// ROLE IN THE PROJECT
//   This is the "ground truth" the GPU result is checked against. It is written
//   to be OBVIOUSLY correct -- a single readable triple loop over voxels, no
//   parallelism, no cleverness -- so that when the GPU and CPU agree, we believe
//   the GPU. The per-(voxel,spot) physics is NOT duplicated here: it is called
//   from the shared proton_physics.h so the CPU and GPU evaluate identical math
//   (docs/PATTERNS.md §2). Compiled by the host C++ compiler only (no CUDA).
//
// READ THIS AFTER: reference_cpu.h, proton_physics.h. Compare to kernels.cu.
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
// read_data_line: pull the next NON-BLANK, NON-COMMENT line from the stream into
// `out`. Lines whose first non-space character is '#' are treated as comments
// and skipped, so the sample file can be self-documenting. Returns false at EOF.
//   WHY a helper: the plan format is line-oriented (grid line, beam line, count,
//   then one line per spot); centralising the "skip comments/blanks" rule keeps
//   the parser below short and robust.
// ---------------------------------------------------------------------------
static bool read_data_line(std::istream& in, std::string& out) {
    std::string line;
    while (std::getline(in, line)) {
        // Find first non-whitespace character to classify the line.
        std::size_t p = line.find_first_not_of(" \t\r\n");
        if (p == std::string::npos) continue;   // blank line -> skip
        if (line[p] == '#') continue;           // comment line -> skip
        out = line;
        return true;
    }
    return false;   // reached end of file
}

Plan load_plan(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open plan file: " + path);

    Plan plan;
    std::string line;

    // ---- line 1: grid geometry --------------------------------------------
    // nx ny nz dx ox oy oz : voxel counts, isotropic spacing, and world origin.
    if (!read_data_line(in, line)) throw std::runtime_error("plan: missing grid line in " + path);
    {
        std::istringstream ss(line);
        Grid& g = plan.grid;
        if (!(ss >> g.nx >> g.ny >> g.nz >> g.dx >> g.ox >> g.oy >> g.oz))
            throw std::runtime_error("plan: bad grid line (need 'nx ny nz dx ox oy oz') in " + path);
        if (g.nx <= 0 || g.ny <= 0 || g.nz <= 0 || g.dx <= 0.0f)
            throw std::runtime_error("plan: grid counts/spacing must be positive in " + path);
    }

    // ---- line 2: beam model + patient surface -----------------------------
    // sigma0 sigma_grow peak_width p_exp z_entry.
    if (!read_data_line(in, line)) throw std::runtime_error("plan: missing beam line in " + path);
    {
        std::istringstream ss(line);
        BeamModel& b = plan.beam;
        if (!(ss >> b.sigma0 >> b.sigma_grow >> b.peak_width >> b.p_exp >> plan.z_entry))
            throw std::runtime_error("plan: bad beam line (need 'sigma0 sigma_grow peak_width p_exp z_entry') in " + path);
        if (b.sigma0 <= 0.0f || b.peak_width <= 0.0f || b.p_exp <= 0.0f)
            throw std::runtime_error("plan: sigma0/peak_width/p_exp must be positive in " + path);
    }

    // ---- line 3: spot count -----------------------------------------------
    int n_spots = 0;
    if (!read_data_line(in, line)) throw std::runtime_error("plan: missing spot count in " + path);
    {
        std::istringstream ss(line);
        if (!(ss >> n_spots) || n_spots <= 0)
            throw std::runtime_error("plan: spot count must be a positive integer in " + path);
    }

    // ---- next n_spots lines: the spots ------------------------------------
    plan.spots.reserve(static_cast<std::size_t>(n_spots));
    for (int s = 0; s < n_spots; ++s) {
        if (!read_data_line(in, line))
            throw std::runtime_error("plan: fewer spot lines than declared in " + path);
        std::istringstream ss(line);
        Spot spot;
        if (!(ss >> spot.x0 >> spot.y0 >> spot.range >> spot.weight))
            throw std::runtime_error("plan: bad spot line (need 'x0 y0 range weight') in " + path);
        plan.spots.push_back(spot);
    }
    return plan;
}

void dose_cpu(const Plan& plan, std::vector<float>& dose) {
    const Grid& g = plan.grid;
    dose.assign(voxel_count(g), 0.0f);            // one dose value per voxel, zeroed

    // Triple loop over voxels (k = depth-ish z, j = y, i = x). For each voxel we
    // sum the contribution of EVERY spot. This nested "for each voxel: for each
    // spot" is the serial form of the GPU's "one thread per voxel loops over
    // spots" -- literally the same arithmetic, just not parallel.
    for (int k = 0; k < g.nz; ++k) {
        // Voxel-centre world z for this depth slice (shared by the whole slice).
        const float vz = g.oz + (static_cast<float>(k) + 0.5f) * g.dx;
        for (int j = 0; j < g.ny; ++j) {
            const float vy = g.oy + (static_cast<float>(j) + 0.5f) * g.dx;
            for (int i = 0; i < g.nx; ++i) {
                const float vx = g.ox + (static_cast<float>(i) + 0.5f) * g.dx;

                // Accumulate this voxel's dose over all spots. We iterate spots
                // in index order so the FP32 summation order is identical to the
                // GPU kernel's -> the two results match tightly (THEORY.md §7).
                float d = 0.0f;
                for (const Spot& s : plan.spots)
                    d += dose_from_spot(plan.beam, s, vx, vy, vz, plan.z_entry);

                dose[voxel_index(g, i, j, k)] = d;
            }
        }
    }
}
