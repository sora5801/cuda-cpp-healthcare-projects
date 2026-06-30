// ===========================================================================
// src/reference_cpu.h  --  Public interface of the CPU reference + data model
// ---------------------------------------------------------------------------
// Project 2.23 : Protein-Ligand Interaction Energy Decomposition
//
// This header is the stable include that main.cu and kernels.cuh pull in to get
// the data model (MmgbsaSystem, ResidueParams, ...) and the CPU-side prototypes
// (load_system, decompose_cpu). The actual per-pair PHYSICS lives in mmgbsa.h
// as `__host__ __device__` inline functions so the CPU reference and the GPU
// kernel run identical math (PATTERNS.md sec 2). We simply re-export it here so
// callers have one obvious include.
//
//   reference_cpu.cpp  -> implements load_system() and decompose_cpu().
//   kernels.cu         -> the GPU twin of decompose_cpu().
//
// READ THIS BEFORE: reference_cpu.cpp, kernels.cuh.
// ===========================================================================
#pragma once

// Pull in the shared data model + the `__host__ __device__` physics core. Every
// type (MmgbsaSystem, PerResidueEnergy, ...) and the load/decompose prototypes
// are declared there; keeping them in mmgbsa.h lets nvcc compile the same
// physics for the device without a separate copy.
#include "mmgbsa.h"
