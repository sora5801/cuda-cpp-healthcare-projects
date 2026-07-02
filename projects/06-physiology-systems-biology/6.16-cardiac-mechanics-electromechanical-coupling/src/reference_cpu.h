// ===========================================================================
// src/reference_cpu.h  --  Ensemble config + CPU reference for the PV-loop sweep
// ---------------------------------------------------------------------------
// Project 6.16 : Cardiac Mechanics & Electromechanical Coupling
//
// The "ensemble" is a 2-D parameter SWEEP over two clinically-meaningful knobs:
//   * CONTRACTILITY  -- the active elastance scale Tref (how stiff/strong the
//                       muscle gets in systole). Reduced Tref models a weak /
//                       failing ventricle.
//   * AFTERLOAD      -- the systemic vascular resistance R_sys (how hard it is
//                       to eject against the arteries). Raised R models
//                       hypertension.
// So the sweep is nT contractility values x nR afterload values = nT*nR virtual
// hearts, each an INDEPENDENT ODE solve -> one GPU thread per heart.
//
// This header holds: the config struct, the (member index -> HeartParams)
// mapping (shared host+device so the kernel reuses it), the text loader, and
// the serial CPU reference. The actual physics/ODE/RK4 is in cardiac.h. Pure
// C++ (no CUDA-only types) so kernels.cu can include it directly.
//
// READ THIS AFTER: cardiac.h. READ THIS BEFORE: reference_cpu.cpp, kernels.cuh.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "cardiac.h"   // CARD_HD, HeartParams, CycleResult, integrate_cycle

// ---------------------------------------------------------------------------
// EnsembleConfig -- everything needed to build + run the whole sweep.
//   The BASELINE HeartParams `base` holds all the fixed physiology; the sweep
//   only overrides Tref (contractility) and R_wind (afterload) per member.
// ---------------------------------------------------------------------------
struct EnsembleConfig {
    HeartParams base;          // baseline physiology (all shared fields)

    double dt_ms = 0.0;        // RK4 timestep [ms]
    int    steps_per_beat = 0; // integration steps per cardiac cycle
    int    n_beats = 0;        // beats to simulate (warm-up + 1 recorded)

    // Sweep grid.
    int    nT = 0, nR = 0;                 // #contractility x #afterload values
    double Tref_lo = 0.0, Tref_hi = 0.0;   // active-elastance range [mmHg/mL]
    double R_lo = 0.0,   R_hi = 0.0;       // afterload (R_sys) range [mmHg*ms/mL]
};

// Number of ensemble members (virtual hearts).
CARD_HD inline int ensemble_size(const EnsembleConfig& c) { return c.nT * c.nR; }

// ---------------------------------------------------------------------------
// member_params -- map a flat member index to that heart's HeartParams.
//   idx = a*nR + b  ->  Tref from row a (nT rows), R_sys from column b (nR).
//   We start from the baseline physiology and override just the two swept
//   fields, so every member differs ONLY in contractility and afterload.
// ---------------------------------------------------------------------------
CARD_HD inline HeartParams member_params(const EnsembleConfig& c, int idx) {
    const int a = idx / c.nR;    // contractility index (0 .. nT-1)
    const int b = idx % c.nR;    // afterload index    (0 .. nR-1)
    HeartParams p = c.base;      // copy baseline, then override two knobs
    p.Tref  = (c.nT > 1) ? c.Tref_lo + (c.Tref_hi - c.Tref_lo) * a / (c.nT - 1)
                         : c.Tref_lo;
    p.R_sys = (c.nR > 1) ? c.R_lo    + (c.R_hi    - c.R_lo)    * b / (c.nR - 1)
                         : c.R_lo;
    return p;
}

// ---------------------------------------------------------------------------
// integrate_member -- run ONE heart to steady state and return its PV summary.
//   Thin wrapper over integrate_cycle() so the CPU loop and the GPU kernel call
//   exactly the same entry point.
// ---------------------------------------------------------------------------
CARD_HD inline CycleResult integrate_member(const EnsembleConfig& c, int idx) {
    const HeartParams p = member_params(c, idx);
    return integrate_cycle(p, c.dt_ms, c.steps_per_beat, c.n_beats);
}

// Load an EnsembleConfig from the text sample (format documented in data/README.md).
EnsembleConfig load_ensemble(const std::string& path);

// CPU reference: integrate every member serially into `results` (sized nT*nR).
// The trusted baseline the GPU ensemble is verified against (same RK4 -> same
// numbers to round-off).
void integrate_cpu(const EnsembleConfig& c, std::vector<CycleResult>& results);
