// ===========================================================================
// src/reference_cpu.h  --  Map model, loader, and CPU reference validation
// ---------------------------------------------------------------------------
// Project 2.22 : Electron Density Map Analysis & Model Validation
//
// WHAT THIS PROJECT COMPUTES
//   Two scores used to VALIDATE a 3-D electron-density map (the kind produced by
//   X-ray crystallography or cryo-EM) against a second map -- either an
//   experimental "half map" or a model-derived map:
//
//     * RSCC  -- Real-Space Correlation Coefficient: a Pearson correlation of the
//                two maps over every voxel. "Does the density agree, voxel by
//                voxel?" Phenix/CCP4 report it per residue for model fit.
//
//     * FSC   -- Fourier Shell Correlation: the normalized cross-correlation of
//                the two maps' Fourier transforms, averaged over spherical shells
//                of spatial frequency |k|. FSC(|k|) is the cryo-EM gold standard
//                for RESOLUTION: the resolution is where FSC first drops below a
//                threshold (0.143 for two independent half-maps).
//
//   The Fourier transform is a Fast Fourier Transform; on the GPU we use the
//   cuFFT library (a batched 3-D real-to-complex FFT) and a per-voxel kernel for
//   the shell binning. The CPU reference uses a naive O(N²)-per-axis DFT -- slow
//   but obviously correct -- so we can verify cuFFT's FFT.
//
// WHY A GPU / cuFFT (the catalog's "Deep dive")
//   Real maps are millions of voxels (256³ ≈ 16.7M); validation FFTs are the
//   bottleneck, and cuFFT does the 3-D transform in O(N log N). The shell sums
//   are an independent per-voxel reduction -- a natural GPU job. This project's
//   lesson is USING A CUDA LIBRARY WITHOUT IT BEING A BLACK BOX (kernels.cu
//   documents exactly what cufftExecR2C computes and its data layout).
//
//   This is a heavily-commented PURE C++ header (no CUDA), so kernels.cu can
//   reuse DensityMap and the loader. Per-voxel math is shared via map_core.h.
//
// READ THIS AFTER: map_core.h (the shared __host__ __device__ formulas).
// Used BY: main.cu, reference_cpu.cpp, kernels.cu.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// DensityMap: a pair of co-sampled cubic electron-density maps on an n×n×n grid.
//   We store BOTH maps a and b together because every computation here compares
//   them on the identical grid. Layout is C-order (z slowest, x fastest):
//       value at (x,y,z) = data[((z*n) + y)*n + x],  0 <= x,y,z < n.
//   `voxel_angstrom` is the physical edge length of one voxel (Å), used only to
//   convert a shell's frequency |k| into a real resolution in Å for reporting.
// ---------------------------------------------------------------------------
struct DensityMap {
    int n = 0;                      // grid edge (cube is n×n×n)
    double voxel_angstrom = 1.0;    // physical voxel size (Å); for resolution labels
    std::vector<float> a;           // [n³] map A (e.g. half-map 1 / experimental)
    std::vector<float> b;           // [n³] map B (e.g. half-map 2 / model)

    long long voxels() const { return static_cast<long long>(n) * n * n; }
};

// ---------------------------------------------------------------------------
// load_map: parse the committed text format (see data/README.md):
//   header line:  "<n> <voxel_angstrom>"
//   then n³ floats for map A, then n³ floats for map B (whitespace-separated).
// Throws std::runtime_error on a bad/truncated file so demos fail loudly.
// ---------------------------------------------------------------------------
DensityMap load_map(const std::string& path);

// ---------------------------------------------------------------------------
// rscc_cpu: the real-space correlation coefficient (Pearson r) of maps a and b
//   over every voxel. The trusted baseline for the GPU's RSCC. Uses the SAME
//   pearson_from_sums() closing formula as the kernel (map_core.h).
// ---------------------------------------------------------------------------
double rscc_cpu(const DensityMap& d);

// ---------------------------------------------------------------------------
// fsc_cpu: the Fourier Shell Correlation curve. Fills `fsc` so that fsc[s] is
//   the correlation in shell s (s = round(|k|)), and `shell_count[s]` is how
//   many reciprocal-space voxels fell in that shell (so main can skip empty
//   shells when picking the resolution). The reference does the 3-D DFT by hand.
//   This is the baseline the cuFFT result is checked against.
// ---------------------------------------------------------------------------
void fsc_cpu(const DensityMap& d, std::vector<double>& fsc,
             std::vector<long long>& shell_count);

// ---------------------------------------------------------------------------
// resolution_at_threshold: the highest-frequency shell index s* at which FSC is
//   still >= `threshold` while every shell up to s* also stayed >= threshold;
//   i.e. the last shell before FSC first crosses below the cutoff. Returns the
//   shell index (an integer frequency). main converts it to Å via shell_to_res.
//   `threshold` is 0.143 for two independent half-maps (the cryo-EM gold std),
//   or 0.5 for map-vs-model. Empty shells (count 0) are skipped.
// ---------------------------------------------------------------------------
int resolution_at_threshold(const std::vector<double>& fsc,
                            const std::vector<long long>& shell_count,
                            double threshold);

// ---------------------------------------------------------------------------
// shell_to_res: convert a shell index (integer spatial frequency, in cycles per
//   box) to a real-space resolution in Å. A box of n voxels at `voxel_angstrom`
//   per voxel spans L = n·voxel_angstrom Å, so a frequency of s cycles-per-box
//   is s/L cycles/Å, i.e. a resolution (period) of L/s = n·voxel/s Å. Shell 0
//   (the DC term) has infinite resolution; we report it as +inf.
// ---------------------------------------------------------------------------
double shell_to_res(int shell, int n, double voxel_angstrom);
