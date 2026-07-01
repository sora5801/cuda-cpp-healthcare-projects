// ===========================================================================
// src/reference_cpu.h  --  CPU reference API + the on-disk data model
// ---------------------------------------------------------------------------
// Project 6.18 : ECG Forward Problem & Body-Surface Potential Mapping
//
// ROLE IN THE PROJECT
//   Declares (a) the ECGData struct that holds a parsed sample file, (b) the
//   loader that fills it, and (c) the plain-C++ reference implementation of the
//   whole forward pipeline. main.cu runs this reference and the GPU kernels and
//   checks that they agree. reference_cpu.cpp implements everything here; the
//   per-entry physics lives in the shared ecg_core.h so CPU and GPU match.
//
// THE PIPELINE IN ONE BREATH
//   1. Build the LEAD-FIELD (transfer) matrix A [L x S]:  A[e][s] = potential at
//      electrode e from a unit-strength dipole source s  (ecg::dipole_potential).
//   2. Apply it to the cardiac source time series X [S x T] to get the
//      BODY-SURFACE POTENTIAL time series  Phi [L x T] = A * X  (a dense GEMM).
//   The GPU accelerates BOTH: a kernel builds A, cuBLAS DGEMM applies it.
//
// READ THIS BEFORE: reference_cpu.cpp, main.cu.  Companion: ecg_core.h.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "ecg_core.h"   // ecg::Vec3, ecg::dipole_potential (shared physics)

// ---------------------------------------------------------------------------
// ECGData: everything one sample file describes, parsed into memory.
//
//   The torso model is a set of L electrodes on the body surface. The cardiac
//   activity is modelled by S fixed current DIPOLES (equivalent-source model):
//   each has a fixed anchor position and a fixed unit direction, but a
//   time-varying STRENGTH that traces out the activation sequence over T frames.
//
//   Matrices are stored ROW-MAJOR in flat vectors (the layout the loader and the
//   kernels agree on):
//     * source_strength : X, shape [S x T], element (s,t) at index s*T + t.
//   Electrode/source geometry is stored as arrays of Vec3.
// ---------------------------------------------------------------------------
struct ECGData {
    int L = 0;   // number of body-surface electrodes (rows of A / Phi)
    int S = 0;   // number of cardiac dipole sources    (cols of A, rows of X)
    int T = 0;   // number of time frames               (cols of X / Phi)

    std::vector<ecg::Vec3> electrode;   // [L] electrode positions (metres)
    std::vector<ecg::Vec3> src_pos;     // [S] dipole anchor positions (metres)
    std::vector<ecg::Vec3> src_dir;     // [S] dipole unit directions (unitless)

    std::vector<double> source_strength;  // X [S*T] row-major: strength(s,t)

    // Demo "ground truth": index of the electrode we EXPECT to record the
    // largest peak-to-peak swing, given how the synthetic sample was built
    // (the electrode nearest the strongest, most-swinging source). A correct
    // pipeline must recover it -- the human-meaningful headline (PATTERNS.md §6).
    int expected_peak_lead = -1;
};

// ---------------------------------------------------------------------------
// load_ecg: parse a sample file into an ECGData (see data/README.md for format).
//   Throws std::runtime_error on any malformed/short file so demos fail loudly
//   rather than silently computing on garbage.
// ---------------------------------------------------------------------------
ECGData load_ecg(const std::string& path);

// ---------------------------------------------------------------------------
// build_lead_field_reference: fill A [L x S] row-major, A[e*S + s] = potential
//   at electrode e from unit-strength source s. Pure serial double loop over
//   ecg::dipole_potential -- the obvious baseline the GPU kernel mirrors.
// ---------------------------------------------------------------------------
void build_lead_field_reference(const ECGData& d, std::vector<double>& A);

// ---------------------------------------------------------------------------
// apply_forward_reference: Phi [L x T] = A [L x S] * X [S x T], all row-major.
//   The triple-loop textbook matrix multiply -- the serial twin of the cuBLAS
//   DGEMM on the GPU. Phi[e][t] is electrode e's potential at time frame t.
// ---------------------------------------------------------------------------
void apply_forward_reference(const std::vector<double>& A,
                             const std::vector<double>& X,
                             int L, int S, int T,
                             std::vector<double>& Phi);
