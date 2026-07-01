// ===========================================================================
// src/dose_problem.h  --  The problem definition shared by every module
// ---------------------------------------------------------------------------
// Project 5.9 -- Gamma-Index Dose Comparison
//
// ROLE IN THE PROJECT
//   A gamma-index comparison needs TWO 2-D dose maps on the SAME grid: the
//   "reference" (usually the treatment-planning-system prediction) and the
//   "evaluated" (usually the measured/QA dose). This struct bundles both maps,
//   the grid geometry (pixel spacing), and the acceptance criteria so that
//   main.cu, reference_cpu.cpp, and kernels.cu all speak the same language.
//
//   It is PURE C++ (no CUDA) so both the host compiler (reference_cpu.cpp) and
//   nvcc (main.cu / kernels.cu) can include it.
//
//   Why 2-D and not the catalog's full 3-D? This is the didactic reduced-scope
//   version (CLAUDE.md §13): a 2-D dose plane is exactly what an IMRT patient-
//   specific QA measurement (film / EPID / 2-D array) produces, it keeps the
//   sample tiny and the math legible, and the extension to 3-D is literally one
//   more nested loop (documented as an exercise + in THEORY §7).
//
// READ THIS BEFORE: reference_cpu.h, kernels.cuh, main.cu.
// ===========================================================================
#pragma once

#include <vector>

// ---------------------------------------------------------------------------
// DoseProblem -- everything needed to run one gamma comparison.
//   The two dose maps are stored row-major (index = y*width + x) as flat
//   vectors, which is exactly the layout the GPU wants for coalesced loads.
// ---------------------------------------------------------------------------
struct DoseProblem {
    int width  = 0;              // grid columns (x)                  [voxels]
    int height = 0;              // grid rows    (y)                  [voxels]
    double spacing_mm = 1.0;     // physical size of one voxel edge   [mm]

    // Acceptance criteria, in human-facing units:
    double dd_percent = 3.0;     // dose-difference criterion         [% of ref max]
    double dta_mm     = 3.0;     // distance-to-agreement criterion   [mm]

    // Only points whose reference dose exceeds this fraction of the max are
    // scored (the "low-dose threshold", standard practice so near-zero
    // background does not dominate the pass-rate). 0.10 == the common 10%.
    double dose_threshold_frac = 0.10;

    // The two dose maps, row-major, length = width*height. Same units on both.
    std::vector<float> ref;      // reference (planned) dose          [dose]
    std::vector<float> eval;     // evaluated (measured) dose         [dose]

    // Convenience: number of voxels in each map.
    int size() const { return width * height; }
};
