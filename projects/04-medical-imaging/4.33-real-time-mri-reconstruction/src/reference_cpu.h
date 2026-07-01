// ===========================================================================
// src/reference_cpu.h  --  Radial-MRI data model, CPU FFT, and CPU gridding recon
// ---------------------------------------------------------------------------
// Project 4.33 : Real-Time MRI Reconstruction
//
// WHAT THIS PROJECT COMPUTES
//   An MRI scanner does NOT measure an image -- it measures k-space, samples of the
//   image's 2D Fourier transform. INTERVENTIONAL / CARDIAC MRI acquires k-space
//   along RADIAL SPOKES through the center at the GOLDEN ANGLE (111.25 deg apart),
//   streaming spoke-by-spoke. Because any contiguous run of golden-angle spokes
//   covers k-space near-uniformly, we can reconstruct a fresh image at every time
//   step from a SLIDING WINDOW of the most recent spokes -- "real-time MRI".
//
//   Radial samples do NOT lie on the Cartesian FFT grid, so we cannot just inverse-
//   FFT them. Instead we do a GRIDDING NUFFT (non-uniform FFT), the workhorse of
//   real-time MRI reconstruction:
//       1. DENSITY-COMPENSATE each sample (radial sampling over-weights the center).
//       2. GRID (convolution-interpolate) every sample onto the Cartesian grid with
//          a small Kaiser-Bessel kernel.
//       3. INVERSE FFT the grid to image space (this is where cuFFT / our radix-2
//          FFT does the heavy lifting).
//       4. DEAPODIZE: divide out the kernel's Fourier transform to correct the
//          brightness roll-off the gridding convolution introduced.
//
// WHY A GPU (the catalog's "cuFFT NUFFT + streaming pipeline" pattern)
//   Real-time cardiac/interventional MRI must reconstruct each frame in <100 ms to
//   guide a catheter or watch a beating heart. A clinical frame is a large grid with
//   many receive coils and thousands of samples per window; the gridding scatter is
//   embarrassingly parallel (one thread per k-space sample) and the FFT is a solved
//   library problem (cuFFT). This teaching project uses a single 2D slice, one coil,
//   and a tiny grid so it runs offline in a fraction of a second -- but the STRUCTURE
//   (density comp -> KB grid -> cuFFT -> deapodize, in a sliding window) is exactly
//   the production one. See THEORY "GPU mapping".
//
//   The GPU path (kernels.cu) grids with an atomic-accumulation kernel and inverse-
//   FFTs with cuFFT; this CPU reference grids with a plain loop and inverse-FFTs with
//   an obviously-correct radix-2 FFT, so the two can be cross-checked. Every per-
//   sample formula both use is imported from grid_core.h.
//
//   Pure C++ header (no CUDA). kernels.cu reuses these structs and grid_core.h.
//
// READ THIS BEFORE: reference_cpu.cpp, main.cu.  READ grid_core.h FIRST.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "grid_core.h"   // Cplx, GriddingParams, kb_weight/deapod, golden_angle

// ---------------------------------------------------------------------------
// RadialData: everything the reconstructor needs, loaded from the sample file.
//   The acquisition is a stack of `n_spokes` radial spokes; each spoke has `n_ro`
//   readout samples spaced along a diameter of k-space through the center. Sample j
//   of spoke s sits at k-space radius (j - n_ro/2) along the spoke's golden angle.
//   We store the measured complex value of every (spoke, readout) sample flat,
//   in spoke-major / readout-minor order: index = s * n_ro + j.
// ---------------------------------------------------------------------------
struct RadialData {
    int n = 0;           // reconstructed image / grid side length (power of two)
    int n_spokes = 0;    // total radial spokes acquired (golden-angle ordered)
    int n_ro = 0;        // readout samples per spoke (along the spoke diameter)
    int win = 0;         // sliding-window size: spokes used per reconstructed frame
    int stride = 0;      // spokes advanced between consecutive frames
    int n_frames = 0;    // number of sliding-window frames to reconstruct
    int kb_w = 4;        // Kaiser-Bessel kernel width in grid cells
    float kb_beta = 0.0f;// Kaiser-Bessel shape parameter (set from kb_w on load)
    std::vector<Cplx> samples;   // [n_spokes * n_ro] measured complex k-space samples
    std::vector<float> truth;    // [n*n] OPTIONAL ground-truth magnitude image (synthetic)
    bool has_truth = false;      // true if the sample carried a ground-truth image

    // params(): bundle the gridding geometry for grid_core.h helpers. kmax = n/2 is
    // the k-space edge in grid units (a full-diameter spoke spans [-n/2, n/2]).
    GriddingParams params() const {
        GriddingParams p;
        p.n = n; p.kb_w = kb_w; p.kb_beta = kb_beta;
        p.kmax = 0.5f * static_cast<float>(n);
        return p;
    }
};

// load_radial: parse the text sample (format documented in data/README.md):
//   line 1: "<n> <n_spokes> <n_ro> <win> <stride> <n_frames> <kb_w> <has_truth>"
//   then n_spokes*n_ro lines "re im" in (spoke-major, readout-minor) order,
//   then (if has_truth) n*n lines "truth" in row-major image order.
// Throws std::runtime_error on a malformed/absent file so demos fail loudly.
RadialData load_radial(const std::string& path);

// ---------------------------------------------------------------------------
// sample_kpos: the Cartesian k-space coordinate (in GRID CELLS, origin at the grid
//   center n/2) of readout sample `j` on spoke `s`. This is the geometry that turns
//   a (spoke, readout) index into an (kx, ky) location the gridder scatters around.
//   Shared by the CPU reference and the GPU kernel so both place samples identically.
//     * s, j   : spoke index and readout index
//     * n_ro   : readout samples per spoke
//     * n      : grid side length (center at n/2)
//     * kx, ky : outputs, k-space position in grid cells (may be fractional)
// ---------------------------------------------------------------------------
void sample_kpos(int s, int j, int n_ro, int n, float& kx, float& ky);

// ---------------------------------------------------------------------------
// fft2_cpu / ifft2_cpu: the trusted CPU reference 2D FFT and its inverse.
//   Separable radix-2 Cooley-Tukey FFT (rows then columns) in DOUBLE precision
//   internally for accuracy, writing back into a Cplx (float) buffer. `fft2_cpu`
//   is the UN-normalized forward transform (matches cuFFT's CUFFT_FORWARD);
//   ifft2_cpu applies the 1/(n*n) normalization so ifft2(fft2(x)) == x. cuFFT
//   likewise leaves its inverse un-normalized, so kernels.cu applies the same scale.
// ---------------------------------------------------------------------------
void fft2_cpu(std::vector<Cplx>& data, int n);        // in-place forward  (no 1/N)
void ifft2_cpu(std::vector<Cplx>& data, int n);       // in-place inverse  (with 1/(n*n))

// ---------------------------------------------------------------------------
// grid_frame_cpu: grid ONE sliding-window frame onto the Cartesian grid (steps 1+2
//   of the NUFFT: density compensation + Kaiser-Bessel convolution). Fills `grid`
//   (size n*n) with the density-compensated, KB-spread complex samples. Spokes used
//   are the contiguous window [spoke0, spoke0 + win).
//     * d      : the loaded radial acquisition
//     * spoke0 : index of the first spoke in this frame's window
//     * grid   : output Cartesian k-space grid (size n*n), zeroed then filled
// ---------------------------------------------------------------------------
void grid_frame_cpu(const RadialData& d, int spoke0, std::vector<Cplx>& grid);

// ---------------------------------------------------------------------------
// reconstruct_frame_cpu: the full CPU gridding reconstruction of ONE frame.
//   Runs grid_frame_cpu -> ifft2_cpu -> deapodize -> magnitude, returning the frame's
//   magnitude image (size n*n). This is the ground truth the GPU frame is checked
//   against; the algorithm is spelled out in reference_cpu.cpp and THEORY.
//     * d       : the loaded acquisition
//     * spoke0  : first spoke of this frame's sliding window
//     * out_mag : filled with the |image| magnitude (size n*n), row-major
// ---------------------------------------------------------------------------
void reconstruct_frame_cpu(const RadialData& d, int spoke0, std::vector<float>& out_mag);

// ---------------------------------------------------------------------------
// kb_beta_for_width: the standard Kaiser-Bessel beta for a given kernel width at
//   2x oversampling (Beatty et al. 2005). We grid onto a grid the same size as the
//   image (no extra oversampling in this teaching version), so we use the widely-
//   cited beta = pi * sqrt(W^2 * (0.5 - 0.5)^2 ... ) simplified to a fixed rule:
//   beta = 2.34 * W works well for W = 4 at modest oversampling. THEORY "the math".
//   Kept here (host-only) so load_radial() and the sample generator agree.
// ---------------------------------------------------------------------------
inline float kb_beta_for_width(int w) {
    // Beatty's formula at oversampling ratio a: beta = pi*sqrt(W^2/a^2*(a-0.5)^2 - 0.8).
    // With a = 2 this reduces to beta = pi*sqrt(W^2*0.5625 - 0.8); for W=4 -> ~7.85.
    const double a = 2.0;
    const double W = static_cast<double>(w);
    double val = 3.14159265358979323846 *
                 std::sqrt(W * W / (a * a) * (a - 0.5) * (a - 0.5) - 0.8);
    if (val < 1.0) val = 1.0;   // guard tiny widths
    return static_cast<float>(val);
}
