# ===========================================================================
# CMakeLists.txt  --  OPTIONAL cross-platform build (the VS .sln is required)
# ---------------------------------------------------------------------------
# Project 1.2 -- Particle-Mesh Ewald Electrostatics
#
# This is a convenience build for Linux/macOS learners and CI. On Windows the
# canonical deliverable is build/particle-mesh-ewald-electrostatics.sln (CLAUDE.md §5). Both
# compile the SAME src/ files, so they stay in lockstep.
#
#   cmake -S . -B build/cmake -DCMAKE_BUILD_TYPE=Release
#   cmake --build build/cmake
# ===========================================================================
cmake_minimum_required(VERSION 3.24)   # 3.24+ for first-class CUDA arch handling
project(particle-mesh-ewald-electrostatics LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# Match the VS project's fat-binary target set (Turing/Ampere/Ada). Override on
# the command line with -DCMAKE_CUDA_ARCHITECTURES=75 for a faster local build.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75 86 89)
endif()

add_executable(particle-mesh-ewald-electrostatics
  src/main.cu
  src/kernels.cu
  src/reference_cpu.cpp
)

# So #include "kernels.cuh", "pme.h" and "util/..." resolve from src/.
target_include_directories(particle-mesh-ewald-electrostatics PRIVATE src)

# This project links cuFFT for the 3D real-to-complex FFT of the charge grid
# (the reciprocal-space step of PME; see src/kernels.cu). CUDA::cufft is the
# imported target the CUDA toolkit provides.
find_package(CUDAToolkit REQUIRED)
target_link_libraries(particle-mesh-ewald-electrostatics PRIVATE CUDA::cufft)
