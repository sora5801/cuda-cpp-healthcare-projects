// ===========================================================================
// src/risk_model.h  --  BEIR-VII lifetime cancer-risk conversion (shared HD)
// ---------------------------------------------------------------------------
// Project 5.10 : Secondary Cancer Risk & Stray-Dose Monte Carlo
//
// WHAT THIS FILE ADDS
//   stray_physics.h gives us a per-organ STRAY DOSE (in fixed-point units). The
//   clinical question, though, is not "how much dose?" but "how much extra cancer
//   risk?" This header converts organ dose -> Excess Lifetime Attributable Risk
//   (LAR) using per-organ risk coefficients in the spirit of the BEIR-VII report
//   (Biological Effects of Ionizing Radiation VII, US National Academies) and
//   ICRP tissue weighting.
//
// THE MODEL (deliberately simple, clearly labelled -- THEORY.md has the caveats)
//   For low doses, LAR is taken proportional to organ equivalent dose H (the
//   Linear No-Threshold, "LNT", assumption used for radiation protection):
//       LAR_organ = risk_coeff_organ * H_organ
//   where risk_coeff_organ (cases per 10^4 persons per sievert) folds in the
//   organ's radiosensitivity, and H_organ is the equivalent dose. Total secondary-
//   cancer LAR is the sum over out-of-field organs. This is exactly the kind of
//   "dose distribution convolved with a lifetime risk model" the catalog calls
//   for; the coefficients here are illustrative teaching values, NOT clinical.
//
// WHY IT IS A SHARED HD FUNCTION
//   We want the CPU reference and any GPU post-processing to compute risk with
//   byte-identical math, so the conversion lives here as an inline HD function
//   (same idiom as stray_physics.h). In this project the risk sum is done on the
//   host from the verified dose tally (it is O(n_organs), negligible), but keeping
//   it HD means a learner can move it onto the GPU as an exercise with zero risk
//   of CPU/GPU divergence.
//
// READ THIS AFTER: stray_physics.h.  READ NEXT: reference_cpu.h.
// ===========================================================================
#pragma once

#include "stray_physics.h"   // HD macro, DOSE_FIXED_SCALE

// Convert a fixed-point dose tally back to a floating "equivalent dose" in the
// same arbitrary teaching unit the physics used (1.0 == one primary-photon energy
// unit). Dividing by DOSE_FIXED_SCALE undoes the fixed-point scaling exactly.
HD inline double fixed_to_dose(unsigned long long fixed) {
    return static_cast<double>(fixed) / DOSE_FIXED_SCALE;
}

// Per-organ excess lifetime attributable risk (LAR), in "cases per 10^4 persons"
// for our unit dose. `risk_coeff` is the organ's BEIR-VII-style sensitivity
// coefficient; `dose_fixed` is that organ's accumulated fixed-point stray dose.
// The Linear No-Threshold assumption makes this a simple product -- the teaching
// point is the *convolution of a dose distribution with a risk model*, not the
// exact coefficient values.
HD inline double organ_lar(double risk_coeff, unsigned long long dose_fixed) {
    return risk_coeff * fixed_to_dose(dose_fixed);
}
