// ===========================================================================
// src/reference_cpu.cpp  --  Loader, shared reduction, serial GIST reference
// ---------------------------------------------------------------------------
// Project 2.26 : Hydrogen Bond Network & Water Placement Analysis
//
// Compiled by the host compiler only. The per-sample physics (water-solute
// energy, voxel lookup) and the per-voxel thermodynamics live in gist.h, shared
// verbatim with the GPU kernel, so this serial baseline and the GPU produce
// identical tallies and an identical ranked hydration-site list. This file owns:
//   * load_dataset()   -- parse the text sample.
//   * derive_voxels()  -- tallies -> sorted VoxelResult list (used by CPU AND GPU).
//   * gist_cpu()       -- the trusted serial scatter-accumulate.
//
// READ THIS AFTER: gist.h, reference_cpu.h.
// ===========================================================================
#include "reference_cpu.h"

#include <algorithm>   // std::sort
#include <fstream>     // std::ifstream
#include <stdexcept>   // std::runtime_error

// ---------------------------------------------------------------------------
// load_dataset: parse the data/sample text format (see data/README.md).
//   HEADER (whitespace-separated):
//     nx ny nz                 voxel counts
//     ox oy oz spacing         grid origin (min corner) + voxel edge (Angstrom)
//     nframes waters_per_frame natoms
//   BODY:
//     natoms lines of: x y z charge          (solute atoms)
//     nframes * waters_per_frame lines of: x y z   (water oxygens, frame-major)
//   We read with operator>> (whitespace-insensitive); make_synthetic.py emits
//   exactly this layout. Any short/garbled field throws so the demo fails loudly.
// ---------------------------------------------------------------------------
Dataset load_dataset(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open dataset file: " + path);

    Dataset d;
    GistGrid& g = d.grid;
    // --- grid geometry ---
    if (!(in >> g.nx >> g.ny >> g.nz))
        throw std::runtime_error("bad header: expected 'nx ny nz' in " + path);
    if (!(in >> g.ox >> g.oy >> g.oz >> g.spacing))
        throw std::runtime_error("bad header: expected 'ox oy oz spacing' in " + path);
    if (g.nx <= 0 || g.ny <= 0 || g.nz <= 0 || g.spacing <= 0.0)
        throw std::runtime_error("non-positive grid dimensions in " + path);
    // --- sizes ---
    if (!(in >> d.nframes >> d.waters_per_frame >> d.natoms))
        throw std::runtime_error("bad header: expected 'nframes waters_per_frame natoms' in " + path);
    if (d.nframes <= 0 || d.waters_per_frame <= 0 || d.natoms <= 0)
        throw std::runtime_error("non-positive counts in " + path);

    // --- solute atoms: x y z charge ---
    d.atoms.resize(static_cast<std::size_t>(d.natoms) * 4);
    for (std::size_t i = 0; i < d.atoms.size(); ++i)
        if (!(in >> d.atoms[i])) throw std::runtime_error("solute atoms truncated in " + path);

    // --- water oxygens: x y z (frame-major) ---
    d.waters.resize(static_cast<std::size_t>(d.num_samples()) * 3);
    for (std::size_t i = 0; i < d.waters.size(); ++i)
        if (!(in >> d.waters[i])) throw std::runtime_error("water records truncated in " + path);

    return d;
}

// ---------------------------------------------------------------------------
// derive_voxels: turn raw per-voxel tallies into a sorted list of hydration
//   sites. Runs once per OCCUPIED voxel (empty voxels carry no water, so they are
//   skipped -- they are not hydration sites). The sort makes stdout deterministic
//   AND puts the most "displaceable" waters (highest GIST dG) first, which is the
//   number a medicinal chemist actually reads.
// ---------------------------------------------------------------------------
std::vector<VoxelResult> derive_voxels(const Dataset& d,
                                       const std::vector<unsigned int>& counts,
                                       const std::vector<gist_fixed_t>& esum) {
    const int nv = d.grid.num_voxels();
    // Under-sampled voxels are noise: a voxel grazed by one stray water has a
    // meaningless mean energy. Require occupancy in at least this many frames
    // before a voxel counts as a hydration SITE (see GIST_MIN_OCCUPANCY_FRACTION).
    const unsigned int min_count = static_cast<unsigned int>(
        GIST_MIN_OCCUPANCY_FRACTION * static_cast<double>(d.nframes));
    std::vector<VoxelResult> voxels;
    voxels.reserve(static_cast<std::size_t>(nv));
    for (int v = 0; v < nv; ++v) {
        const unsigned int c = counts[static_cast<std::size_t>(v)];
        if (c < min_count || c == 0) continue;        // unoccupied / under-sampled
        voxels.push_back(gist_voxel_result(v, c, esum[static_cast<std::size_t>(v)],
                                           d.nframes, d.grid));
    }
    // Deterministic ranking. We rank by OCCUPANCY (count) first: just like
    // WaterMap/GIST, the pipeline first IDENTIFIES hydration sites as the voxels
    // where waters cluster most persistently (the occupancy map), then annotates
    // each with its thermodynamics. Occupancy is the robust, low-noise signal; the
    // per-voxel mean energy of a sparsely-visited voxel is statistically shaky, so
    // letting raw dG drive the ranking would surface noise. Ties in count are
    // broken by dG descending (the displaceability score), then by voxel index --
    // a strict total order, so std::sort is reproducible across platforms.
    std::sort(voxels.begin(), voxels.end(),
              [](const VoxelResult& a, const VoxelResult& b) {
                  if (a.count != b.count) return a.count > b.count;   // occupancy first
                  if (a.dG != b.dG)       return a.dG > b.dG;         // then displaceability
                  return a.index < b.index;                          // then index (tiebreak)
              });
    return voxels;
}

// ---------------------------------------------------------------------------
// gist_cpu: serial scatter-accumulate, the trusted baseline.
//   For each (frame, water): compute its voxel and its water-solute energy, then
//   ADD into that voxel's count and fixed-point energy sum. Identical math to the
//   GPU kernel -- only the loop (serial here, one-thread-per-sample there) differs.
//   Fixed-point energy means the serial sum and the parallel atomic sum are the
//   SAME integer regardless of order, so verification is exact.
// ---------------------------------------------------------------------------
std::vector<VoxelResult> gist_cpu(const Dataset& d,
                                  std::vector<unsigned int>& counts,
                                  std::vector<gist_fixed_t>& esum) {
    const int nv = d.grid.num_voxels();
    counts.assign(static_cast<std::size_t>(nv), 0u);
    esum.assign(static_cast<std::size_t>(nv), 0);

    const float* atoms = d.atoms.data();
    for (int f = 0; f < d.nframes; ++f) {
        for (int w = 0; w < d.waters_per_frame; ++w) {
            // Flat offset of this water's (x,y,z) in the frame-major array.
            const std::size_t base =
                (static_cast<std::size_t>(f) * d.waters_per_frame + w) * 3;
            const double wx = d.waters[base + 0];
            const double wy = d.waters[base + 1];
            const double wz = d.waters[base + 2];

            const int v = gist_voxel_of(d.grid, wx, wy, wz);
            if (v < 0) continue;                       // water outside the grid box

            // Water<->solute interaction energy (kcal/mol), quantized to fixed-point.
            const double e = gist_water_solute_energy(wx, wy, wz, atoms, d.natoms);
            counts[static_cast<std::size_t>(v)] += 1u;
            esum[static_cast<std::size_t>(v)]   += gist_to_fixed(e);
        }
    }
    return derive_voxels(d, counts, esum);
}
