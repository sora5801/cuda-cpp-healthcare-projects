# ===========================================================================
# CMakeLists.txt  --  OPTIONAL cross-platform build (the VS .sln is required)
# ---------------------------------------------------------------------------
# Project 2.22 -- Electron Density Map Analysis & Model Validation
#
# This is a convenience build for Linux/macOS learners and CI. On Windows the
# canonical deliverable is build/electron-density-map-analysis-model-validation.sln (CLAUDE.md §5). Both
# compile the SAME src/ files, so they stay in lockstep.
#
#   cmake -S . -B build/cmake -DCMAKE_BUILD_TYPE=Release
#   cmake --build build/cmake
# ===========================================================================
cmake_minimum_required(VERSION 3.24)   # 3.24+ for first-class CUDA arch handling
project(electron-density-map-analysis-model-validation LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# Match the VS project's fat-binary target set (Turing/Ampere/Ada). Override on
# the command line with -DCMAKE_CUDA_ARCHITECTURES=75 for a faster local build.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75 86 89)
endif()

add_executable(electron-density-map-analysis-model-validation
  src/main.cu
  src/kernels.cu
  src/reference_cpu.cpp
)

# So #include "kernels.cuh" and "util/..." resolve from src/.
target_include_directories(electron-density-map-analysis-model-validation PRIVATE src)

# This project links the cuFFT library for the 3-D forward FFT (see kernels.cu).
find_package(CUDAToolkit REQUIRED)
target_link_libraries(electron-density-map-analysis-model-validation PRIVATE CUDA::cufft)
