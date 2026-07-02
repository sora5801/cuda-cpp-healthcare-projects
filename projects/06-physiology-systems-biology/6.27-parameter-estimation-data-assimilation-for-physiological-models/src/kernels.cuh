// ===========================================================================
// src/kernels.cuh  --  GPU ensemble-forecast interface
// ---------------------------------------------------------------------------
// Project 6.27 : Parameter Estimation & Data Assimilation for Physiological Models
//
// THE BIG IDEA (pattern: ENSEMBLE ODE FORECAST + host-side EnKF analysis)
//   The Ensemble Kalman Filter alternates two steps per observation:
//     FORECAST  -- integrate EVERY ensemble member forward one window. Members are
//                  independent, so this is embarrassingly parallel: one GPU thread
//                  per member runs the whole RK4 window in registers. The catalog
//                  calls this "the bottleneck" -- it is what we accelerate.
//     ANALYSIS  -- a tiny dense-linear-algebra correction using the ensemble's own
//                  sample covariance. For our 3-vector state it is a few scalar
//                  sums, done on the HOST (enkf_analysis in reference_cpu.cpp); the
//                  SAME function serves the CPU and GPU paths (docs/PATTERNS.md §5).
//
//   forecast_gpu integrates one window for all members on the device. run_enkf_gpu
//   drives the whole filter: it loops windows, calling forecast_gpu then the shared
//   host enkf_analysis, so the GPU result is a true twin of run_enkf_cpu and they
//   match to round-off.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, windkessel.h, reference_cpu.h.
// ===========================================================================
#pragma once

#include <vector>

#include "reference_cpu.h"   // EnKFConfig, EnKFResult (pure C++, safe in a .cu)

// Device kernel: thread `idx` forecasts ensemble member idx by one window.
//   ens : device pointer to the [m * WK_NSTATE] ensemble (row-major), updated in place
//   Everything else is passed by value so each thread reads it from a register.
__global__ void forecast_kernel(double* __restrict__ ens, int m,
                                double t0, double dt, int substeps,
                                double T, double t_sys, double Q_peak);

// Host wrapper: forecast ALL members one window on the GPU (H2D, launch, D2H).
//   Updates `ensemble` in place. Adds the measured kernel time to *accum_ms so the
//   driver can report the total forecast cost across all windows.
void forecast_gpu(const EnKFConfig& c, std::vector<double>& ensemble, double t0,
                  float* accum_ms);

// Drive the WHOLE EnKF on the GPU forecast path: forecast (device) + analysis
//   (shared host fn) for every window. Mirrors run_enkf_cpu exactly, so their
//   final ensembles agree to ~round-off. *forecast_ms returns the summed kernel time.
EnKFResult run_enkf_gpu(const EnKFConfig& c, const std::vector<double>& observations,
                        std::vector<double>& ensemble_out, float* forecast_ms);
