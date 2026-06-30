// ===========================================================================
// src/deer_params.h  --  Fixed problem geometry shared by host + device code.
// ---------------------------------------------------------------------------
// Project 2.35 : Electron Paramagnetic Resonance (EPR/DEER) Constrained Modeling
//
// These compile-time constants define the DEER distance axis (the histogram
// grid) and the spin-label rotamer cloud size. They live in their own tiny
// header (no functions, no CUDA) so that EVERY translation unit -- the GPU
// kernels, the CPU reference, and main -- sees the SAME grid. Changing the grid
// in one place changes it everywhere, which is exactly what we want for a
// CPU-vs-GPU comparison that must line up bin-for-bin.
//
// Units: all distances are in NANOMETRES (nm), the natural unit for DEER (the
// technique routinely resolves 1.5-8 nm separations).
//
// READ THIS BEFORE: deer.h (it uses every constant here).
// ===========================================================================
#pragma once

// --- The DEER distance axis (the P(r) histogram grid) ----------------------
// We tile the experimentally accessible window [R_MIN_NM, R_MIN_NM+NBINS*BIN)
// into NBINS uniform bins. 0.1 nm bins over 1.5-6.5 nm (50 bins) is a realistic
// resolution for a typical DEER reconstruction.
static constexpr int    NBINS    = 50;      // number of distance bins in P(r)
static constexpr double R_MIN_NM = 1.5;     // lower edge of the first bin (nm)
static constexpr double R_BIN_NM = 0.1;     // bin width (nm)  -> covers up to 6.5 nm

// --- Spin-label rotamer cloud ----------------------------------------------
// Each engineered site carries a flexible nitroxide label (MTSSL) that samples
// many rotameric states. We approximate the rotamer library by ROTAMERS_PER_SITE
// equally weighted endpoints per site per frame. The back-calculation convolves
// the two clouds: ROTAMERS_PER_SITE^2 spin pairs per frame. A real MTSSL library
// (e.g. MMM's) has ~200 Boltzmann-weighted states; 24 keeps the demo fast while
// still producing a realistically broad single-frame P(r).
static constexpr int ROTAMERS_PER_SITE = 24;

// --- Reweighting solver knobs (used by main / reference / kernels) ---------
// REWEIGHT_ITERS gradient-descent steps over the log-weights; REWEIGHT_LR is the
// step size. Both are fixed (not adaptive) so the run is fully deterministic and
// the CPU and GPU take exactly the same trajectory. THETA is the BioEn/EROS
// confidence parameter balancing fit (chi^2) against the entropy regularizer.
static constexpr int    REWEIGHT_ITERS = 4000;     // fixed-length descent -> deterministic
static constexpr double REWEIGHT_LR    = 50.0;     // learning rate on the log-weights
static constexpr double THETA          = 1.0e-4;   // entropy weight (small => trust the data more)
