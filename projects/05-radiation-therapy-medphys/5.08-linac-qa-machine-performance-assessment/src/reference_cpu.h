// ===========================================================================
// src/reference_cpu.h  --  CPU reference: gamma index + linac QA metrics
// ---------------------------------------------------------------------------
// Project 5.8 : Linac QA & Machine Performance Assessment  (catalog ID 5.8)
//
// WHY A SEPARATE PURE-C++ HEADER
//   reference_cpu.cpp is compiled by the plain host compiler (cl.exe / g++) and
//   must never see CUDA syntax (__global__ etc.), so its prototypes cannot live
//   in kernels.cuh. main.cu and reference_cpu.cpp both include THIS header so
//   they agree on the data types and function signatures. The per-pixel gamma
//   math they share lives in gamma.h (the __host__ __device__ core).
//
// WHAT THIS PROJECT COMPUTES  (see README §"What this computes" and THEORY.md)
//   A linac quality-assurance (QA) workflow on a 2-D dose plane:
//     (A) 2-D GAMMA INDEX of a MEASURED (EPID / portal-dosimetry) dose plane
//         against the PLANNED (reference) dose plane, yielding a per-pixel gamma
//         map and the clinical gamma PASS RATE (TG-218: >= 95% at 3%/3mm).
//     (B) MACHINE-PERFORMANCE METRICS from the measured plane's central axis
//         profiles: central-axis (CAX) output, beam FLATNESS and SYMMETRY over
//         the flattened region -- the daily-QA numbers a physicist checks.
//   The GPU accelerates the gamma map (kernels.cu); everything is verified
//   against this CPU reference (main.cu). The QA scorecard is deterministic.
//
// READ THIS BEFORE: reference_cpu.cpp (implements these), main.cu (calls them).
// After: gamma.h (the shared per-pixel math), then kernels.cuh for the GPU twin.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "gamma.h"   // GammaParams + gamma_value_at (shared host/device core)

// ---------------------------------------------------------------------------
// QAProblem -- everything loaded from a sample file, in one bundle.
//   The two dose planes are stored row-major, length ny*nx each, in the SAME
//   arbitrary dose units. `norm_dose` is the normalisation dose used to turn the
//   percentage dose-difference criterion into absolute dose units (global gamma).
// ---------------------------------------------------------------------------
struct QAProblem {
    int   nx = 0;             // plane width  (columns)
    int   ny = 0;             // plane height (rows)
    float spacing_mm = 1.0f;  // pixel size, mm (square pixels)
    float dd_percent = 3.0f;  // dose-difference criterion, % of norm_dose
    float dta_mm     = 3.0f;  // distance-to-agreement criterion, mm
    float norm_dose  = 0.0f;  // normalisation dose (0 => use the plane's max)
    std::vector<float> meas;  // measured / delivered plane  [ny*nx]
    std::vector<float> ref;   // reference / planned plane   [ny*nx]
};

// ---------------------------------------------------------------------------
// load_qa: parse a whitespace sample file into a QAProblem.
//   File layout (all whitespace-separated; see data/sample & make_synthetic.py):
//       nx ny spacing_mm dd_percent dta_mm norm_dose
//       <nx*ny reference-plane values, row-major>
//       <nx*ny measured-plane  values, row-major>
//   Throws std::runtime_error if the file is missing or malformed so demos fail
//   loudly instead of silently running on garbage.
// ---------------------------------------------------------------------------
QAProblem load_qa(const std::string& path);

// ---------------------------------------------------------------------------
// make_gamma_params: derive the GammaParams (used by BOTH CPU and GPU) from a
//   QAProblem. Converts dd_percent (% of the normalisation dose) into absolute
//   dose units and picks a pixel search radius that safely covers the DTA.
// ---------------------------------------------------------------------------
GammaParams make_gamma_params(const QAProblem& q);

// ---------------------------------------------------------------------------
// gamma_map_cpu: the CPU reference gamma computation.
//   Loops every measured pixel and calls gamma_value_at (gamma.h) -- the exact
//   same function the GPU kernel calls -- filling `gamma_out` (resized ny*nx).
//   This is the trusted baseline the GPU result is checked against.
// ---------------------------------------------------------------------------
void gamma_map_cpu(const QAProblem& q, const GammaParams& p,
                   std::vector<float>& gamma_out);

// ---------------------------------------------------------------------------
// gamma_pass_rate: fraction (in percent) of EVALUATED pixels with gamma <= 1.
//   Only pixels whose measured dose is at or above `dose_threshold` are counted
//   (the standard "low-dose threshold" that excludes near-zero background where
//   gamma is meaningless). Deterministic integer counting -> exact.
// ---------------------------------------------------------------------------
float gamma_pass_rate(const QAProblem& q, const std::vector<float>& gamma_map,
                      float pass_gamma, float dose_threshold,
                      int& n_eval, int& n_pass);

// ---------------------------------------------------------------------------
// QAMetrics -- the machine-performance numbers derived from the MEASURED plane's
//   central-axis profiles (one horizontal, one vertical through the plane centre).
// ---------------------------------------------------------------------------
struct QAMetrics {
    float cax_dose;       // dose at the central axis (plane centre)
    float flatness_pct;   // (Dmax - Dmin)/(Dmax + Dmin) * 100 over the flat region
    float symmetry_pct;   // max |D(+x) - D(-x)| / CAX * 100 over the flat region
    float field_width_mm; // FWHM of the horizontal profile (50% of CAX), in mm
};

// ---------------------------------------------------------------------------
// compute_qa_metrics: flatness / symmetry / output / field size from the
//   measured plane. The "flat region" is the central 80% of the field width
//   (the standard convention). Pure host math; deterministic.
// ---------------------------------------------------------------------------
QAMetrics compute_qa_metrics(const QAProblem& q);
