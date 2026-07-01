// ===========================================================================
// src/reference_cpu.cpp  --  Loader + serial ensemble integration (the baseline)
// ---------------------------------------------------------------------------
// Project 5.12 : FLASH Radiotherapy GPU Modeling
//
// ROLE IN THE PROJECT
//   This is the "ground truth" the GPU result is checked against. It is written
//   to be OBVIOUSLY correct -- a single readable loop over ensemble members, no
//   parallelism, no cleverness -- so that when the GPU and CPU agree, we believe
//   the GPU. The per-voxel chemistry (integrate_voxel) lives in flash.h and is
//   shared verbatim with the kernel, so agreement is exact to round-off.
//
//   Compiled by the host C++ compiler only (no CUDA here). See reference_cpu.h.
//
// READ THIS AFTER: flash.h, reference_cpu.h. Compare against kernels.cu (GPU twin).
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>
#include <stdexcept>

// Parse the whitespace-separated ensemble configuration file. The format is
// fixed and documented in data/README.md; we read the fields in order and
// validate the ones that would otherwise cause a divide-by-zero or an empty run.
EnsembleConfig load_ensemble(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open ensemble file: " + path);
    EnsembleConfig c;
    if (!(in >> c.total_dose >> c.n_pulses >> c.dt >> c.conv_steps_per_gap
             >> c.flash_steps_per_gap >> c.relax_steps
             >> c.n_po2 >> c.po2_lo >> c.po2_hi))
        throw std::runtime_error("bad parameters (expected 'total_dose n_pulses dt "
            "conv_steps_per_gap flash_steps_per_gap relax_steps n_po2 po2_lo po2_hi') in " + path);
    if (c.total_dose <= 0 || c.n_pulses <= 0 || c.dt <= 0 || c.conv_steps_per_gap <= 0 ||
        c.flash_steps_per_gap <= 0 || c.n_po2 <= 0)
        throw std::runtime_error("invalid ensemble parameters (non-positive) in " + path);
    return c;
}

// CPU reference: integrate every ensemble member serially. `results` is sized to
// ensemble_size(c). Each member is an independent per-voxel chemistry solve, so
// this is a plain loop here and one GPU thread per member in kernels.cu. The
// (idx -> VoxelJob) mapping and the ODE are both shared (reference_cpu.h /
// flash.h), so this loop and the kernel do byte-identical arithmetic.
void integrate_cpu(const EnsembleConfig& c, std::vector<VoxelResult>& results) {
    const int M = ensemble_size(c);
    results.assign(static_cast<std::size_t>(M), VoxelResult{});
    for (int idx = 0; idx < M; ++idx) {
        const VoxelJob j = member_job(c, idx);
        results[static_cast<std::size_t>(idx)] = integrate_voxel(j);
    }
}
