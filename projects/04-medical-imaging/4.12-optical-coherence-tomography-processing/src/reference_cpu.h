// ===========================================================================
// src/reference_cpu.h  --  OCT data model + CPU reconstruction reference
// ---------------------------------------------------------------------------
// Project 4.12 : Optical Coherence Tomography Processing (SD-OCT reconstruction)
//
// WHAT THIS PROJECT COMPUTES
//   Spectral-domain OCT (SD-OCT) shines low-coherence light into tissue and reads
//   the interference SPECTRUM of light back-scattered from every depth against a
//   reference mirror. One such spectrum is an "A-scan" (a single axial line). The
//   depth profile of reflectivity is recovered by:
//       (1) removing the DC/background,
//       (2) windowing the spectrum (suppress FFT side lobes),
//       (3) DISPERSION COMPENSATION (a k-dependent phase correction), and
//       (4) an inverse FFT along the spectral axis -> reflectivity vs. depth.
//   Stacking many adjacent A-scans across the lateral direction gives a 2-D
//   cross-section image: a "B-scan". This project reconstructs a whole B-scan.
//
// WHY A GPU / cuFFT (the catalog's GPU pattern)
//   A single B-scan is thousands of A-scans, each needing a length-N FFT, and the
//   A-scans are INDEPENDENT. That is exactly the batched-FFT shape cuFFT is built
//   for: one cufftExecC2C call does the entire B-scan's FFTs at once. Real-time
//   3-D OCT (surgical guidance) processes ~100 B-scans/s -- billions of FFT
//   points per second -- feasible only on the GPU. This project teaches the
//   "USE A CUDA LIBRARY WITHOUT IT BEING A BLACK BOX" lesson (cuFFT) PLUS a custom
//   kernel (dispersion compensation) that the library cannot do for us.
//
//   (Per the catalog, the full pipeline also includes k-space resampling/NUFFT
//   and CNN layer segmentation. This is the REDUCED-SCOPE teaching version -- the
//   spectral reconstruction core -- with the rest described in THEORY "real
//   world". See CLAUDE.md §13.)
//
//   This is a pure C++ header (no CUDA), so kernels.cu can reuse OctBscan and the
//   host compiler can compile reference_cpu.cpp. Per-sample math lives in
//   oct_core.h (shared by CPU + GPU).
//
// READ THIS AFTER: oct_core.h. READ THIS BEFORE: kernels.cuh, main.cu.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// OctBscan: one B-scan of raw SD-OCT spectra + reconstruction parameters.
//   Layout is row-major with the spectral sample as the fast axis:
//       raw[a * n_spec + i] = spectral sample i of A-scan a.
//   n_spec MUST be even (we keep the physical depth range 0..N/2).
// ---------------------------------------------------------------------------
struct OctBscan {
    int    n_ascan = 0;      // number of A-scans (lateral pixels) in the B-scan
    int    n_spec  = 0;      // spectral samples per A-scan == FFT length N
    double a2 = 0.0;         // 2nd-order dispersion coefficient (see oct_core.h)
    double a3 = 0.0;         // 3rd-order dispersion coefficient
    std::vector<float> raw;  // [n_ascan * n_spec] raw interferometric spectra
};

// Depth samples kept in the reconstructed image. A real-valued spectrum has a
// Hermitian-symmetric FFT, so the second half of the depth profile is a mirror
// image (the "complex-conjugate artifact"). We keep only the physical first half
// [0 .. N/2). This helper expresses that convention in code.
inline int oct_depth_count(int n_spec) { return n_spec / 2; }

// ---------------------------------------------------------------------------
// load_bscan(path): parse the text format documented in data/README.md:
//     header:  "<n_ascan> <n_spec> <a2> <a3>"
//     then n_ascan rows, each of n_spec whitespace-separated raw spectrum values.
//   Throws std::runtime_error on any malformed / truncated input so demos fail
//   loudly rather than silently reconstructing garbage.
// ---------------------------------------------------------------------------
OctBscan load_bscan(const std::string& path);

// ---------------------------------------------------------------------------
// reconstruct_cpu(b, image): the TRUSTED reference reconstruction.
//   For each A-scan: compute the DC (mean), run preprocess_sample() over the
//   spectrum (oct_core.h), take a NAIVE O(N^2) DFT, keep the first N/2 depth bins
//   as normalised power (0..1 vs the A-scan's own peak). Output:
//       image[a * (N/2) + z] = normalised linear reflectivity power at depth z.
//   O(N^2) per A-scan -- slow but obviously correct; cuFFT reproduces it O(N logN).
// ---------------------------------------------------------------------------
void reconstruct_cpu(const OctBscan& b, std::vector<double>& image);

// ---------------------------------------------------------------------------
// peak_depths(image, n_ascan, n_depth, out): the deterministic, integer result
//   we print and verify. For each A-scan, the depth bin (argmax) of the strongest
//   reflector. Integer indices are order-independent, so this is byte-identical
//   between CPU and GPU and reproducible run to run (PATTERNS.md #3).
// ---------------------------------------------------------------------------
void peak_depths(const std::vector<double>& image, int n_ascan, int n_depth,
                 std::vector<int>& out);
