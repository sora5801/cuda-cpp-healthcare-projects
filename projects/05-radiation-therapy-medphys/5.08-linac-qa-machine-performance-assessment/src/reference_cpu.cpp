// ===========================================================================
// src/reference_cpu.cpp  --  Trusted CPU baseline for the linac-QA workflow
// ---------------------------------------------------------------------------
// Project 5.8 : Linac QA & Machine Performance Assessment  (catalog ID 5.8)
//
// ROLE
//   The readable, single-threaded reference implementation of everything the
//   program computes: (A) the 2-D gamma-index map + pass rate, and (B) the
//   beam flatness/symmetry/output metrics. The GPU path (kernels.cu) reproduces
//   ONLY the gamma map; main.cu checks the two agree bit-for-bit (both call the
//   same gamma_value_at() from gamma.h). The metrics are host-only teaching
//   extras that turn a bare gamma number into a realistic QA scorecard.
//
//   Compiled by the plain host C++ compiler -> this file must contain NO CUDA.
//
// READ THIS AFTER: reference_cpu.h (the contract) and gamma.h (the shared math).
// ===========================================================================
#include "reference_cpu.h"

#include <algorithm>   // std::max, std::min, std::max_element
#include <cmath>       // std::fabs, std::sqrt
#include <stdexcept>   // std::runtime_error

#include "util/io.hpp" // util::read_floats

// ---------------------------------------------------------------------------
// load_qa: read the header line + the two planes from a whitespace text file.
//   We reuse util::read_floats (slurps every number) and then interpret the
//   flat vector by the documented layout. Being strict about the count catches
//   truncated files early.
// ---------------------------------------------------------------------------
QAProblem load_qa(const std::string& path) {
    const std::vector<float> v = util::read_floats(path);   // throws if missing
    // Header is 6 numbers; guard before indexing.
    if (v.size() < 6) {
        throw std::runtime_error("QA sample too short (need a 6-number header): " + path);
    }
    QAProblem q;
    q.nx         = static_cast<int>(v[0]);
    q.ny         = static_cast<int>(v[1]);
    q.spacing_mm = v[2];
    q.dd_percent = v[3];
    q.dta_mm     = v[4];
    q.norm_dose  = v[5];
    if (q.nx <= 0 || q.ny <= 0) {
        throw std::runtime_error("QA sample has non-positive plane dimensions: " + path);
    }
    const size_t n = static_cast<size_t>(q.nx) * q.ny;   // pixels per plane
    // We need the header (6) + reference plane (n) + measured plane (n).
    if (v.size() < 6 + 2 * n) {
        throw std::runtime_error("QA sample has fewer values than nx*ny*2 + header: " + path);
    }
    q.ref.assign (v.begin() + 6,     v.begin() + 6 + n);       // planned first
    q.meas.assign(v.begin() + 6 + n, v.begin() + 6 + 2 * n);   // measured second

    // If no explicit normalisation dose was given (0), use the reference plane's
    // maximum -- the usual "global gamma normalised to the max planned dose".
    if (q.norm_dose <= 0.0f) {
        q.norm_dose = *std::max_element(q.ref.begin(), q.ref.end());
    }
    return q;
}

// ---------------------------------------------------------------------------
// make_gamma_params: package the tolerances + geometry for gamma_value_at().
//   Key conversion: dd_percent is a PERCENT of norm_dose; the per-pixel math
//   works in absolute dose units, so dd = dd_percent/100 * norm_dose. The search
//   radius (in pixels) must cover a few DTA so the true nearest reference point
//   is never missed -- we use ceil(3*DTA / spacing) as a safe, cheap bound.
// ---------------------------------------------------------------------------
GammaParams make_gamma_params(const QAProblem& q) {
    GammaParams p;
    p.nx         = q.nx;
    p.ny         = q.ny;
    p.spacing_mm = q.spacing_mm;
    p.dd         = q.dd_percent * 0.01f * q.norm_dose;  // % -> absolute dose units
    p.dta_mm     = q.dta_mm;
    // Look out to ~3 DTA in each direction. Beyond that the space term alone
    // (dist/DTA)^2 already exceeds 9 >> 1, so those points cannot be the minimum
    // for any passing pixel. ceil via +0.999f keeps it an integer pixel count.
    p.search_radius = static_cast<int>(3.0f * q.dta_mm / q.spacing_mm + 0.999f);
    if (p.search_radius < 1) p.search_radius = 1;
    p.pass_gamma = 1.0f;
    return p;
}

// ---------------------------------------------------------------------------
// gamma_map_cpu: fill gamma_out[y*nx + x] = gamma(measured pixel (x,y)).
//   A plain doubly-nested loop over the plane, delegating each pixel to the
//   shared gamma_value_at(). O(nx*ny * (2R+1)^2) work; the GPU version does the
//   identical arithmetic with one thread per measured pixel.
// ---------------------------------------------------------------------------
void gamma_map_cpu(const QAProblem& q, const GammaParams& p,
                   std::vector<float>& gamma_out) {
    gamma_out.assign(static_cast<size_t>(q.nx) * q.ny, 0.0f);
    for (int my = 0; my < q.ny; ++my) {
        for (int mx = 0; mx < q.nx; ++mx) {
            gamma_out[static_cast<size_t>(my) * q.nx + mx] =
                gamma_value_at(q.meas.data(), q.ref.data(), mx, my, p);
        }
    }
}

// ---------------------------------------------------------------------------
// gamma_pass_rate: percentage of evaluated pixels that pass (gamma <= threshold).
//   "Evaluated" = measured dose >= dose_threshold (skip near-zero background).
//   Counting is pure integer arithmetic, so the rate is deterministic and does
//   not depend on summation order (unlike a float accumulation).
// ---------------------------------------------------------------------------
float gamma_pass_rate(const QAProblem& q, const std::vector<float>& gamma_map,
                      float pass_gamma, float dose_threshold,
                      int& n_eval, int& n_pass) {
    n_eval = 0;
    n_pass = 0;
    const size_t n = static_cast<size_t>(q.nx) * q.ny;
    for (size_t i = 0; i < n; ++i) {
        if (q.meas[i] < dose_threshold) continue;   // low-dose cut
        ++n_eval;
        if (gamma_map[i] <= pass_gamma) ++n_pass;
    }
    // Return percent; guard the (degenerate) empty-evaluation case.
    return (n_eval > 0) ? (100.0f * static_cast<float>(n_pass) / static_cast<float>(n_eval))
                        : 0.0f;
}

// ---------------------------------------------------------------------------
// Local helper: extract the horizontal central-axis profile (the middle row of
// the measured plane). Returned by value; small (nx floats).
// ---------------------------------------------------------------------------
static std::vector<float> central_row(const QAProblem& q) {
    const int cy = q.ny / 2;                     // central-axis row
    std::vector<float> row(q.nx);
    for (int x = 0; x < q.nx; ++x) {
        row[x] = q.meas[static_cast<size_t>(cy) * q.nx + x];
    }
    return row;
}

// ---------------------------------------------------------------------------
// compute_qa_metrics: flatness, symmetry, CAX output, and field width from the
//   MEASURED plane's horizontal central-axis profile.
//
//   Definitions (standard AAPM conventions, taught in THEORY.md §"QA metrics"):
//     * CAX output   = dose at the plane centre.
//     * field width  = FWHM: distance between the two points where the profile
//                      crosses 50% of the CAX dose (the beam edges).
//     * flat region  = central 80% of the field width (edges/penumbra excluded).
//     * flatness     = (Dmax - Dmin)/(Dmax + Dmin) * 100  over the flat region.
//     * symmetry     = max over the flat region of |D(+x) - D(-x)| / CAX * 100,
//                      i.e. the worst left/right imbalance about the centre.
// ---------------------------------------------------------------------------
QAMetrics compute_qa_metrics(const QAProblem& q) {
    QAMetrics m{};
    const std::vector<float> row = central_row(q);
    const int nx = q.nx;
    const int cx = nx / 2;                        // central-axis column index
    m.cax_dose = row[cx];

    // --- Field edges via the 50%-of-CAX crossings (FWHM). ------------------
    // Walk outward from the centre in each direction until we drop below 50%.
    const float half = 0.5f * m.cax_dose;
    int left = 0, right = nx - 1;
    for (int x = cx; x >= 0; --x)      { if (row[x] < half) { left  = x; break; } }
    for (int x = cx; x < nx; ++x)      { if (row[x] < half) { right = x; break; } }
    const int field_pix = right - left;                       // FWHM in pixels
    m.field_width_mm = static_cast<float>(field_pix) * q.spacing_mm;

    // --- Flat region = central 80% of the FWHM. ---------------------------
    // margin = 10% of the field on each side (so 80% remains in the middle).
    const int margin = static_cast<int>(0.1f * static_cast<float>(field_pix) + 0.5f);
    const int flo = cx - (field_pix / 2 - margin);            // flat-region start
    const int fhi = cx + (field_pix / 2 - margin);            // flat-region end
    const int lo = std::max(0, flo);
    const int hi = std::min(nx - 1, fhi);

    // --- Flatness = (max-min)/(max+min) over the flat region. -------------
    float dmax = row[lo], dmin = row[lo];
    for (int x = lo; x <= hi; ++x) {
        dmax = std::max(dmax, row[x]);
        dmin = std::min(dmin, row[x]);
    }
    m.flatness_pct = (dmax + dmin > 0.0f)
                   ? (dmax - dmin) / (dmax + dmin) * 100.0f
                   : 0.0f;

    // --- Symmetry = worst |D(+x) - D(-x)| / CAX over the flat region. ------
    float worst = 0.0f;
    const int reach = field_pix / 2 - margin;                 // pixels each side
    for (int d = 0; d <= reach; ++d) {
        const int xr = cx + d, xl = cx - d;
        if (xl < 0 || xr >= nx) break;
        const float asym = std::fabs(row[xr] - row[xl]);
        if (asym > worst) worst = asym;
    }
    m.symmetry_pct = (m.cax_dose > 0.0f) ? (worst / m.cax_dose * 100.0f) : 0.0f;

    return m;
}
