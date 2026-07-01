// ===========================================================================
// src/reference_cpu.h  --  PK/PD population config loader + CPU reference
// ---------------------------------------------------------------------------
// Project 6.15 : PK/PD & PBPK Modeling
//
// Pure C++ (no CUDA). The model + RK4 + per-patient sampling live in pkpd.h;
// kernels.cu reuses PkPdParams + PatientResult from there too. This CPU
// reference integrates the IDENTICAL virtual patients as the GPU (shared RNG +
// RK4), so the per-patient PK/PD metrics match to round-off -> exact verification
// (PATTERNS.md §2, §4).
//
// READ THIS AFTER: pkpd.h.  READ THIS BEFORE: reference_cpu.cpp, main.cu.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "pkpd.h"   // PkPdParams, PatientResult, pkpd_integrate

// Load PkPdParams from the whitespace-separated one-line text format
// (documented in data/README.md):
//   "dose ka CL Vc kin kout Imax IC50 cv dt steps n_patients seed"
// Throws std::runtime_error on a missing file or malformed / non-physical values
// so demos fail loudly instead of silently running on garbage.
PkPdParams load_pkpd(const std::string& path);

// CPU reference: integrate every virtual patient SERIALLY. `results` is resized
// to n_patients. This is the trusted baseline the GPU population is checked
// against in main.cu -- a plain loop over the same pkpd_integrate() the kernel
// calls per thread.
void integrate_cpu(const PkPdParams& P, std::vector<PatientResult>& results);
