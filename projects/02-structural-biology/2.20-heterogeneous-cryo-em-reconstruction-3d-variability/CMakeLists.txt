# ===========================================================================
# CMakeLists.txt  --  OPTIONAL cross-platform build (the VS .sln is required)
# ---------------------------------------------------------------------------
# Project 2.20 -- Heterogeneous Cryo-EM Reconstruction (3D Variability)   (template skeleton)
#
# This is a convenience build for Linux/macOS learners and CI. On Windows the
# canonical deliverable is build/heterogeneous-cryo-em-reconstruction-3d-variability.sln (CLAUDE.md §5). Both
# compile the SAME src/ files, so they stay in lockstep.
#
#   cmake -S . -B build/cmake -DCMAKE_BUILD_TYPE=Release
#   cmake --build build/cmake
# ===========================================================================
cmake_minimum_required(VERSION 3.24)   # 3.24+ for first-class CUDA arch handling
project(heterogeneous-cryo-em-reconstruction-3d-variability LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# Match the VS project's fat-binary target set (Turing/Ampere/Ada). Override on
# the command line with -DCMAKE_CUDA_ARCHITECTURES=75 for a faster local build.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75 86 89)
endif()

add_executable(heterogeneous-cryo-em-reconstruction-3d-variability
  src/main.cu
  src/kernels.cu
  src/reference_cpu.cpp
)

# So #include "kernels.cuh" and "util/..." resolve from src/.
target_include_directories(heterogeneous-cryo-em-reconstruction-3d-variability PRIVATE src)

# This project links cuSOLVER for the dense symmetric eigendecomposition of the
# N x N Gram matrix (Dsyevd; see src/kernels.cu). cuBLAS/cuSPARSE are its
# transitive dependencies (PATTERNS.md §5; BUILD_GUIDE §7b).
find_package(CUDAToolkit REQUIRED)
target_link_libraries(heterogeneous-cryo-em-reconstruction-3d-variability
  PRIVATE CUDA::cusolver CUDA::cublas CUDA::cusparse)
