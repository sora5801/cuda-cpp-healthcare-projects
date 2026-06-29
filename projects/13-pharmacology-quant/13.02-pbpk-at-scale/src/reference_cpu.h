// ===========================================================================
// src/reference_cpu.h  --  PBPK population config + CPU reference
// ---------------------------------------------------------------------------
// Project 13.02 : PBPK at Scale
//
// Pure C++ (no CUDA). The model + RK4 + sampling are in pbpk.h; kernels.cu reuses
// PbpkParams + PatientResult. The CPU reference integrates the identical patients
// as the GPU (shared RNG + RK4), so the per-patient results match to round-off.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "pbpk.h"   // PbpkParams, PatientResult, pbpk_integrate

// Load PbpkParams from the one-line text format (data/README.md):
//   "dose ka CL Vc Vp Q cv dt steps n_patients seed"
PbpkParams load_pbpk(const std::string& path);

// CPU reference: integrate every virtual patient serially. results sized to
// n_patients. The trusted baseline the GPU population is checked against.
void integrate_cpu(const PbpkParams& P, std::vector<PatientResult>& results);
