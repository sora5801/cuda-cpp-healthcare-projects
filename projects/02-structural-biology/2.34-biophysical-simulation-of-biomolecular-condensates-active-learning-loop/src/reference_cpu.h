// ===========================================================================
// src/reference_cpu.h  --  Ensemble config, CPU reference, and the AL acquisition
// ---------------------------------------------------------------------------
// Project 2.34 : Biophysical Simulation of Biomolecular Condensates
//                (Active Learning Loop)  --  reduced-scope teaching version
//
// WHAT THIS HEADER DECLARES
//   * EnsembleConfig    -- the whole experiment: the CG model constants plus the
//                          stickiness (lambda) sweep that defines the candidate
//                          sequences. Reused unchanged by the GPU kernel.
//   * load_ensemble     -- parse the one-line text config (see data/README.md).
//   * integrate_cpu     -- the trusted serial baseline: run every replica with
//                          the SAME shared integrator (condensate.h) the GPU
//                          uses, so GPU == CPU within a documented tolerance.
//   * acquisition_score / propose_next_lambda -- the ACTIVE-LEARNING step: a
//                          cheap deterministic surrogate + acquisition function
//                          that picks the next sequence to simulate. The reduced
//                          stand-in for the GNN surrogate + Bayesian optimization
//                          loop the catalog describes (THEORY).
//
// The actual physics/integrator lives in condensate.h (shared host+device).
// This file is pure C++ (no CUDA), so kernels.cu can include it safely.
//
// READ THIS AFTER: condensate.h.   READ THIS BEFORE: kernels.cuh, main.cu.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "condensate.h"   // CondensateModel, ReplicaResult, integrate_replica, cohesive_lambda

// ---------------------------------------------------------------------------
// EnsembleConfig: one fully-specified active-learning iteration.
//   The candidate sequences are reduced to a uniform grid of n_members
//   stickiness values in [lambda_lo, lambda_hi]; member m simulates lambda(m)
//   (cohesive_lambda() in condensate.h). k_cohese is the base cohesive stiffness
//   scale shared by all replicas. target_D is the experimental diffusion
//   coefficient we want to MATCH by sequence design -- the acquisition function
//   proposes the lambda whose measured D is closest to it.
// ---------------------------------------------------------------------------
struct EnsembleConfig {
    CondensateModel model;          // CG-MD constants (beads, steps, dt, kT, ...)
    int    n_members = 0;           // number of candidate sequences (ensemble size)
    double lambda_lo = 0.0;         // low end of the stickiness scan
    double lambda_hi = 0.0;         // high end of the stickiness scan
    double k_cohese  = 0.0;         // base cohesive stiffness scale (all replicas)
    double target_D  = 0.0;         // experimental target diffusion coefficient
};

// Number of ensemble members (one trajectory each). Tiny accessor so the kernel
// and the host agree on the count. CND_HD (host+device) because the GUARD in the
// kernel reads it on the device too (condensate.h defines CND_HD).
CND_HD inline int ensemble_size(const EnsembleConfig& c) { return c.n_members; }

// Map member m to its stickiness lambda on the sweep grid (thin wrapper over
// cohesive_lambda so callers don't repeat the lo/hi/size arguments). CND_HD so
// the kernel can pick each thread's lambda directly on the device.
CND_HD inline double member_lambda(const EnsembleConfig& c, int m) {
    return cohesive_lambda(m, c.n_members, c.lambda_lo, c.lambda_hi);
}

// ---------------------------------------------------------------------------
// load_ensemble: read the one-line text config (format in data/README.md):
//   n_beads steps dt kT gamma k_bond r0 eq_steps seed
//   n_members lambda_lo lambda_hi k_cohese target_D
// Throws std::runtime_error on a malformed/short file so demos fail loudly.
// ---------------------------------------------------------------------------
EnsembleConfig load_ensemble(const std::string& path);

// ---------------------------------------------------------------------------
// integrate_cpu: the serial reference. Runs every replica with integrate_replica
// (the same function the GPU kernel calls) and fills results[m]. This is BOTH
// the teaching baseline (its wall time makes the GPU speed-up legible) and the
// ground truth the GPU ensemble is checked against. results is sized n_members.
// ---------------------------------------------------------------------------
void integrate_cpu(const EnsembleConfig& c, std::vector<ReplicaResult>& results);

// ---------------------------------------------------------------------------
// THE ACTIVE-LEARNING STEP (deterministic, host-side reduction over the result)
// ---------------------------------------------------------------------------
// acquisition_score: rank a candidate by how promising it is to pursue NEXT.
//   In a real loop a GNN surrogate predicts property(sequence) WITH uncertainty
//   and Bayesian optimization maximizes an acquisition (e.g. Expected Improvement
//   or Upper Confidence Bound) to trade exploitation vs exploration. Here the
//   "surrogate" is the just-measured ensemble itself, and the objective is to
//   MATCH a target diffusion coefficient, so the score is simply |D - target|.
//   SMALLER is better (it is a residual). Deterministic -> reproducible pick.
//   measured_D : this member's diffusion coefficient (length^2 / time)
//   target_D   : the experimental value we want to match
//   ret        : the acquisition residual (smaller = more promising)
double acquisition_score(double measured_D, double target_D);

// propose_next_lambda: the loop's output. Given the finished ensemble, return
// the lambda (and its member index via *best_member) minimizing acquisition_score
// -- the sequence Bayesian optimization would simulate at higher fidelity next.
// Deterministic argmin (ties broken by lowest member index).
double propose_next_lambda(const EnsembleConfig& c,
                           const std::vector<ReplicaResult>& results,
                           int* best_member);
