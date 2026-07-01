// ===========================================================================
// src/reference_cpu.h  --  Model loader + CPU reference knockout screen
// ---------------------------------------------------------------------------
// Project 6.12 : Metabolic Flux / Constraint-Based Modeling
//
// This header declares the HOST-side pieces:
//   * load_model()   -- parse a metabolic model from the text format below into
//                        an FbaModel (the LP defined in fba.h).
//   * screen_cpu()   -- the trusted CPU reference: solve the wild-type LP plus
//                        every single-reaction knockout, serially. The GPU kernel
//                        (kernels.cu) computes the SAME array; main.cu compares
//                        them element-by-element.
//
// The actual LP solver + the FbaModel/FbaResult structs live in fba.h (shared
// host+device). This file is pure C++ (no CUDA), so reference_cpu.cpp compiles
// under cl.exe / g++ and kernels.cu can also #include it for the struct + loader.
//
// TEXT MODEL FORMAT (see data/README.md for the annotated sample):
//   Lines beginning with '#' and blank lines are ignored, EXCEPT a line whose
//   first token is exactly "#names:" supplies the nrxn reaction labels.
//   The remaining tokens are read positionally as:
//     nmet nrxn
//     S[0,0] ... S[0,nrxn-1]     (row 0 of the stoichiometry matrix)
//     ...                        (nmet rows total, row-major)
//     lb[0]  ... lb[nrxn-1]      (lower flux bounds)
//     ub[0]  ... ub[nrxn-1]      (upper flux bounds)
//     c[0]   ... c[nrxn-1]       (objective coefficients; biomass reaction = 1)
//
// READ THIS AFTER: fba.h. Then reference_cpu.cpp, kernels.cuh, main.cu.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "fba.h"   // FbaModel, FbaResult, solve_fba, solve_knockout (host+device)

// Load an FbaModel (and reaction names) from the text format above.
//   path  : file to read.
//   names : filled with nrxn labels -- from a "#names:" line if present,
//           otherwise "R0","R1",... Throws std::runtime_error on any parse or
//           capacity error so the demo fails loudly instead of on garbage input.
FbaModel load_model(const std::string& path, std::vector<std::string>& names);

// CPU reference: solve the wild-type LP and every single-reaction knockout.
//   model   : the loaded FBA model.
//   results : filled with (nrxn + 1) FbaResults. Index k in [0,nrxn) is the
//             knockout of reaction k; the LAST entry (index nrxn) is the wild
//             type (no deletion). This layout matches the GPU output exactly.
// Serial by construction -- the teaching baseline the GPU screen is checked
// against. Because both call the identical solve_knockout()/solve_fba() from
// fba.h, the two result arrays agree bit-for-bit (deterministic simplex).
void screen_cpu(const FbaModel& model, std::vector<FbaResult>& results);
