// ===========================================================================
// src/reference_cpu.h  --  Ensemble config + CPU reference integration
// ---------------------------------------------------------------------------
// Project 5.12 : FLASH Radiotherapy GPU Modeling
//
// The ensemble is a 2-D sweep:  n_po2 oxygen levels  x  2 delivery modes
// (CONVENTIONAL, FLASH) = n_po2*2 independent per-voxel chemistry solves. The
// config, the (member index -> VoxelJob) mapping, and the file format live here
// (all shared host+device so the GPU kernel reuses them); the actual ODE/RK4 is
// in flash.h. This header is pure C++ (no CUDA-only types), so kernels.cu can
// safely include it and reuse EnsembleConfig.
//
// READ THIS AFTER: flash.h (the physics). READ BEFORE: kernels.cuh, main.cu.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "flash.h"   // FLASH_HD, VoxelJob, VoxelResult, integrate_voxel, FlashRates

// The two delivery modes we contrast. Same TOTAL dose; they differ only in how
// the pulses are spaced in time (pulse_gap_s), which is the whole point.
enum DeliveryMode { MODE_CONVENTIONAL = 0, MODE_FLASH = 1, N_MODES = 2 };

// One ensemble job: fixed beam/integration settings + the pO2 sweep range. The
// two delivery modes are implicit (every pO2 is run in BOTH modes). The ONLY
// difference between the modes is how many RK4 sub-steps are integrated between
// consecutive pulse onsets: CONVENTIONAL uses a LARGE number (a long inter-pulse
// gap, so O2 refills and radicals clear before the next pulse); FLASH uses a
// TINY number (UHDR -- pulses stack before O2 can recover). The physical gap
// duration is (steps_per_gap * dt) seconds; we store it too, for reporting.
struct EnsembleConfig {
    double total_dose = 0.0;    // total prescribed dose [Gy] (split over pulses)
    int    n_pulses   = 0;      // pulses in the train (dose_per_pulse=total/n_pulses)
    double dt         = 0.0;    // RK4 timestep [s]
    int    conv_steps_per_gap = 0;  // RK4 sub-steps between pulses, CONVENTIONAL (large)
    int    flash_steps_per_gap = 0; // RK4 sub-steps between pulses, FLASH/UHDR (tiny)
    int    relax_steps   = 0;   // post-delivery relaxation sub-steps (both modes)
    int    n_po2 = 0;           // number of oxygen levels swept
    double po2_lo = 0.0, po2_hi = 0.0;  // pO2 sweep range [mmHg]
};

// Number of ensemble members = (oxygen levels) x (2 delivery modes).
FLASH_HD inline int ensemble_size(const EnsembleConfig& c) { return c.n_po2 * N_MODES; }

// Map a flat member index to the (pO2 level, delivery mode) it represents.
//   idx = po2_index * N_MODES + mode.  So consecutive pairs share a pO2 and
//   differ only by mode -- convenient for the FLASH-vs-CONVENTIONAL comparison.
FLASH_HD inline void member_axes(const EnsembleConfig& c, int idx,
                                 int& po2_index, int& mode) {
    po2_index = idx / N_MODES;
    mode      = idx % N_MODES;
    (void)c;                    // c is unused here but kept for a uniform signature
}

// Build the concrete VoxelJob for ensemble member `idx`. This is the ONE place
// that turns a config + index into the per-voxel parameters both CPU and GPU
// integrate, so they cannot drift apart.
FLASH_HD inline VoxelJob member_job(const EnsembleConfig& c, int idx) {
    int po2_index, mode;
    member_axes(c, idx, po2_index, mode);

    // pO2 sampled linearly across [po2_lo, po2_hi]; guard n_po2==1.
    const double frac = (c.n_po2 > 1) ? (double)po2_index / (c.n_po2 - 1) : 0.0;
    const double po2  = c.po2_lo + (c.po2_hi - c.po2_lo) * frac;

    // The delivery mode selects the inter-pulse step count (FLASH = tiny gap,
    // CONVENTIONAL = large gap). Everything else -- dose, pulses, chemistry -- is
    // identical, so any difference in the result is due to timing alone.
    const int spg = (mode == MODE_FLASH) ? c.flash_steps_per_gap
                                         : c.conv_steps_per_gap;

    VoxelJob j;
    j.po2_mmHg       = po2;
    j.dose_per_pulse = c.total_dose / c.n_pulses;   // same total dose either mode
    j.n_pulses       = c.n_pulses;
    j.pulse_gap_s    = spg * c.dt;                   // physical inter-pulse gap [s]
    j.dt             = c.dt;
    j.steps_per_gap  = spg;
    j.relax_steps    = c.relax_steps;
    j.k              = default_rates();
    return j;
}

// Load an EnsembleConfig from the whitespace text format (see data/README.md):
//   "total_dose n_pulses dt conv_steps_per_gap flash_steps_per_gap relax_steps
//    n_po2 po2_lo po2_hi"
EnsembleConfig load_ensemble(const std::string& path);

// CPU reference: integrate every member serially into `results` (sized to
// ensemble_size). This is the trusted baseline the GPU ensemble is checked
// against -- same integrate_voxel() -> same numbers to round-off.
void integrate_cpu(const EnsembleConfig& c, std::vector<VoxelResult>& results);
