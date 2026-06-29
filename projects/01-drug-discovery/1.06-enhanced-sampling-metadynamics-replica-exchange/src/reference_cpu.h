// ===========================================================================
// src/reference_cpu.h  --  Ensemble config + CPU reference (multi-walker MetaD)
// ---------------------------------------------------------------------------
// Project 1.6 : Enhanced Sampling -- Metadynamics & Replica Exchange
//
// The "ensemble" here is a set of independent metadynamics WALKERS, all sampling
// the same double-well landscape but with different RNG streams (different walker
// ids) and alternating start wells. Each walker is a fully independent run of
// metad::run_walker(): a plain loop on the CPU here, one GPU thread each in
// kernels.cu. The per-walker physics lives in metad.h so CPU and GPU agree
// exactly. This header defines the run configuration, the text-file loader, and
// the serial reference integration. Pure C++ -- kernels.cu reuses MetadConfig.
//
// READ THIS AFTER: metad.h. Compare reference_cpu.cpp against kernels.cu.
// ===========================================================================
#pragma once

#include <cstdint>   // uint64_t
#include <string>
#include <vector>

#include "metad.h"   // metad::Model, run_walker, recover_fes, WalkerResult

// One metadynamics ensemble job: the physical model + how many walkers to run.
//   We embed metad::Model (the per-walker parameters) and add the ensemble size
//   plus the starting CV value. Every walker shares the same Model; only its RNG
//   stream (its id) and its start well differ.
struct MetadConfig {
    metad::Model model;   // landscape + thermostat + metadynamics + grid settings
    int      n_walkers;   // number of independent walkers in the ensemble
    uint64_t seed;        // base RNG seed (walker id mixes in for an independent stream)
    double   s_start;     // |start| CV magnitude; walkers alternate +/- to seed both wells
};

// Number of walkers in the ensemble (mirrors the GPU thread count).
// METAD_HD (from metad.h) makes it callable from BOTH host code and the kernel.
METAD_HD inline int ensemble_size(const MetadConfig& c) { return c.n_walkers; }

// The deterministic start position of walker `id`: even ids start in the left
// well (-s_start), odd ids in the right well (+s_start). Seeding both wells makes
// the recovered FES symmetric and the demo result interpretable. Shared host+
// device so the kernel and the CPU reference compute identical start positions.
METAD_HD inline double walker_start(const MetadConfig& c, int id) {
    return (id % 2 == 0) ? -c.s_start : +c.s_start;
}

// Load a MetadConfig from the whitespace-separated text format (see data/README.md):
//   A kT mass friction dt steps hill_w hill_sigma deposit_every bias_factor
//   s_lo s_hi nbins n_walkers seed s_start
// Throws std::runtime_error on a missing file or malformed/invalid parameters.
MetadConfig load_config(const std::string& path);

// CPU reference: run every walker serially with metad::run_walker(), collecting
// each walker's WalkerResult and accumulating the ENSEMBLE-AVERAGE bias grid
// (the average over walkers' bias grids -- the multi-walker FES estimate).
//   results   : sized to n_walkers, one summary per walker.
//   mean_bias : sized to nbins, the average bias grid across all walkers.
// This is the trusted baseline the GPU run is checked against (identical math).
void integrate_cpu(const MetadConfig& c,
                   std::vector<metad::WalkerResult>& results,
                   std::vector<double>& mean_bias);
