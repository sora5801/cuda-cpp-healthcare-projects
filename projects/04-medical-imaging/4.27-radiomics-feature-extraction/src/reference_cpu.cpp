// ===========================================================================
// src/reference_cpu.cpp  --  Loader, shared reductions, serial CPU reference
// ---------------------------------------------------------------------------
// Project 4.27 : Radiomics Feature Extraction
//
// ROLE IN THE PROJECT
//   This is the "ground truth" the GPU result is checked against. The GLCM
//   builder here is a single readable serial triple loop over voxels x the 13
//   directions -- obviously correct, no parallelism. When the GPU's atomic
//   scatter reproduces these exact integer counts, we believe the GPU.
//
//   Compiled by the host C++ compiler only (no CUDA here). The per-voxel math
//   (quantize, direction offsets, indexing) is shared with the GPU via
//   radiomics.h, and the GLCM->features + histogram->features reductions in this
//   file are ALSO called by the GPU wrapper (kernels.cu) so both paths report
//   byte-identical features. See reference_cpu.h.
//
// READ THIS AFTER: radiomics.h, reference_cpu.h. Compare build_glcm_cpu() with
// the atomic build_glcm kernel in kernels.cu (the GPU twin).
// ===========================================================================
#include "reference_cpu.h"

#include <cmath>       // std::log2, std::sqrt
#include <fstream>
#include <limits>
#include <stdexcept>

// ---------------------------------------------------------------------------
// load_volume: parse the tiny text format described in data/README.md.
//   Header: "nx ny nz Ng"; then nx*ny*nz intensities (row-major, x fastest);
//   then nx*ny*nz mask flags (0/1). We also scan the ROI to find the intensity
//   range [vmin, vmax] used for quantization, and count ROI voxels (nroi).
// ---------------------------------------------------------------------------
Volume load_volume(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open volume file: " + path);

    Volume v;
    if (!(in >> v.nx >> v.ny >> v.nz >> v.Ng)
        || v.nx <= 0 || v.ny <= 0 || v.nz <= 0 || v.Ng <= 1 || v.Ng > 256)
        throw std::runtime_error("bad header (expected 'nx ny nz Ng') in " + path);

    const std::size_t n = v.voxels();
    v.intensity.resize(n);
    for (std::size_t i = 0; i < n; ++i)
        if (!(in >> v.intensity[i])) throw std::runtime_error("intensities truncated in " + path);

    v.mask.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        int m;
        if (!(in >> m)) throw std::runtime_error("mask truncated in " + path);
        v.mask[i] = (m != 0) ? 1u : 0u;
    }

    // Scan the ROI for its intensity range and voxel count. Quantization uses
    // this same [vmin,vmax] on both CPU and GPU -> identical gray levels.
    float lo = std::numeric_limits<float>::infinity();
    float hi = -std::numeric_limits<float>::infinity();
    int count = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (!v.mask[i]) continue;
        lo = std::fmin(lo, v.intensity[i]);
        hi = std::fmax(hi, v.intensity[i]);
        ++count;
    }
    if (count == 0) throw std::runtime_error("ROI mask is empty (no voxels) in " + path);
    v.vmin = lo;
    v.vmax = hi;
    v.nroi = count;
    return v;
}

// ---------------------------------------------------------------------------
// build_histogram: count how many ROI voxels fall in each gray level [0,Ng-1].
//   This is a plain histogram (order-independent), the basis of first-order
//   features. Integer counts -> exact and trivially CPU/GPU-matching.
// ---------------------------------------------------------------------------
void build_histogram(const Volume& v, std::vector<long long>& hist) {
    hist.assign(v.Ng, 0);
    const std::size_t n = v.voxels();
    for (std::size_t i = 0; i < n; ++i) {
        if (!v.mask[i]) continue;                         // outside ROI: skip
        const int g = rad_quantize(v.intensity[i], v.vmin, v.vmax, v.Ng);
        ++hist[g];
    }
}

// ---------------------------------------------------------------------------
// first_order_from_histogram: mean / variance / energy / entropy of the ROI
//   gray-level distribution. p[g] = hist[g] / nroi is the discrete PDF.
//     mean     = sum g * p[g]
//     variance = sum (g-mean)^2 * p[g]
//     energy   = sum hist[g] * g^2         (raw sum of squares, IBSI "energy")
//     entropy  = -sum p[g] log2 p[g]        (bits; 0 for a single level)
// ---------------------------------------------------------------------------
void first_order_from_histogram(const Volume& v, const std::vector<long long>& hist,
                                Features& f) {
    const double n = static_cast<double>(v.nroi);
    double mean = 0.0, energy = 0.0, entropy = 0.0;
    for (int g = 0; g < v.Ng; ++g) {
        const double c = static_cast<double>(hist[g]);
        const double p = c / n;                            // probability of level g
        mean   += g * p;
        energy += c * static_cast<double>(g) * g;          // integer sum of squares
        if (p > 0.0) entropy -= p * std::log2(p);          // 0*log0 := 0
    }
    double variance = 0.0;
    for (int g = 0; g < v.Ng; ++g) {
        const double p = static_cast<double>(hist[g]) / n;
        const double d = static_cast<double>(g) - mean;
        variance += d * d * p;
    }
    f.mean = mean;
    f.variance = variance;
    f.energy = energy;
    f.entropy = entropy;
}

// ---------------------------------------------------------------------------
// haralick_from_glcm: turn a direction-summed COUNT matrix into probabilities
//   and read off the classic Haralick scalars.
//
//   The count matrix passed in has already been SYMMETRIZED on the counting side
//   (each pair contributed to both (i,j) and (j,i)); we simply normalize by the
//   grand total to get P(i,j) with sum P = 1. Then:
//     contrast    = sum P(i,j) (i-j)^2
//     energy(ASM) = sum P(i,j)^2
//     homogeneity = sum P(i,j) / (1 + (i-j)^2)
//     entropy     = -sum P(i,j) log2 P(i,j)
//     correlation = (sum i*j*P - mu_i mu_j) / (sig_i sig_j)   in [-1, 1]
//   where mu/sig are the means/std-devs of the GLCM marginal row and column
//   distributions (equal here because the matrix is symmetric).
// ---------------------------------------------------------------------------
void haralick_from_glcm(const Volume& v, const std::vector<long long>& glcm, Features& f) {
    const int Ng = v.Ng;

    // Grand total of co-occurrence counts (integer -> exact). If zero (e.g. a
    // 1-voxel ROI with no neighbours), features stay at their zero defaults.
    long long total = 0;
    for (int i = 0; i < Ng; ++i)
        for (int j = 0; j < Ng; ++j)
            total += glcm[rad_glcm_index(i, j, Ng)];
    f.glcm_total = total;
    if (total == 0) return;
    const double inv_total = 1.0 / static_cast<double>(total);

    // Marginal mean of the row index i under P (== column mean, by symmetry).
    double mu = 0.0;
    for (int i = 0; i < Ng; ++i) {
        double row_p = 0.0;
        for (int j = 0; j < Ng; ++j)
            row_p += static_cast<double>(glcm[rad_glcm_index(i, j, Ng)]) * inv_total;
        mu += i * row_p;
    }
    // Marginal variance of i (== of j).
    double var = 0.0;
    for (int i = 0; i < Ng; ++i) {
        double row_p = 0.0;
        for (int j = 0; j < Ng; ++j)
            row_p += static_cast<double>(glcm[rad_glcm_index(i, j, Ng)]) * inv_total;
        const double d = static_cast<double>(i) - mu;
        var += d * d * row_p;
    }
    const double sig = std::sqrt(var);

    double contrast = 0.0, asm_energy = 0.0, homog = 0.0, entropy = 0.0, corr_num = 0.0;
    for (int i = 0; i < Ng; ++i) {
        for (int j = 0; j < Ng; ++j) {
            const double p = static_cast<double>(glcm[rad_glcm_index(i, j, Ng)]) * inv_total;
            if (p <= 0.0) continue;                        // skip empty cells (also 0*log0)
            const double diff = static_cast<double>(i) - static_cast<double>(j);
            contrast   += p * diff * diff;
            asm_energy += p * p;
            homog      += p / (1.0 + diff * diff);
            entropy    -= p * std::log2(p);
            corr_num   += (static_cast<double>(i) - mu) * (static_cast<double>(j) - mu) * p;
        }
    }
    f.glcm_contrast    = contrast;
    f.glcm_energy      = asm_energy;
    f.glcm_homogeneity = homog;
    f.glcm_entropy     = entropy;
    // Correlation is undefined when the gray levels never vary (sig == 0);
    // report the conventional value 1 (perfectly correlated constant field).
    f.glcm_correlation = (sig > 0.0) ? (corr_num / (sig * sig)) : 1.0;
}

// ---------------------------------------------------------------------------
// build_glcm_cpu: the serial reference GLCM builder.
//   For every ROI voxel and each of the 13 directions, if the neighbour is in
//   bounds AND in the ROI, increment BOTH (i,j) and (j,i) (symmetrization). The
//   result is the direction-summed, symmetric COUNT matrix the GPU must match.
//   Complexity: O(nroi * 13) integer increments.
// ---------------------------------------------------------------------------
void build_glcm_cpu(const Volume& v, std::vector<long long>& glcm) {
    glcm.assign(static_cast<std::size_t>(v.Ng) * v.Ng, 0);

    RadDir dirs[RAD_NUM_DIRECTIONS];
    rad_fill_directions(dirs);                             // same 13 offsets as the GPU

    for (int z = 0; z < v.nz; ++z) {
        for (int y = 0; y < v.ny; ++y) {
            for (int x = 0; x < v.nx; ++x) {
                const std::size_t idx = rad_vox_index(x, y, z, v.nx, v.ny);
                if (!v.mask[idx]) continue;                // centre must be in ROI
                const int gi = rad_quantize(v.intensity[idx], v.vmin, v.vmax, v.Ng);

                for (int k = 0; k < RAD_NUM_DIRECTIONS; ++k) {
                    const int xn = x + dirs[k].dx;
                    const int yn = y + dirs[k].dy;
                    const int zn = z + dirs[k].dz;
                    // Bounds check: neighbours off the grid do not co-occur.
                    if (xn < 0 || xn >= v.nx || yn < 0 || yn >= v.ny || zn < 0 || zn >= v.nz)
                        continue;
                    const std::size_t nidx = rad_vox_index(xn, yn, zn, v.nx, v.ny);
                    if (!v.mask[nidx]) continue;           // neighbour must be in ROI too
                    const int gj = rad_quantize(v.intensity[nidx], v.vmin, v.vmax, v.Ng);

                    // Symmetric increment: count the pair in both orders so the
                    // matrix is symmetric (direction & its opposite merged).
                    ++glcm[rad_glcm_index(gi, gj, v.Ng)];
                    ++glcm[rad_glcm_index(gj, gi, v.Ng)];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// extract_features_cpu: the full reference pipeline.
// ---------------------------------------------------------------------------
Features extract_features_cpu(const Volume& v) {
    Features f;
    std::vector<long long> hist;
    build_histogram(v, hist);
    first_order_from_histogram(v, hist, f);

    std::vector<long long> glcm;
    build_glcm_cpu(v, glcm);
    haralick_from_glcm(v, glcm, f);
    return f;
}
