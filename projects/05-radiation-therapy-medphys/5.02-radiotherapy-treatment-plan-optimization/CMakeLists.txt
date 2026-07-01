# ===========================================================================
# CMakeLists.txt  --  OPTIONAL cross-platform build (the VS .sln is required)
# ---------------------------------------------------------------------------
# Project 5.2 -- Radiotherapy Treatment-Plan Optimization   (template skeleton)
#
# This is a convenience build for Linux/macOS learners and CI. On Windows the
# canonical deliverable is build/radiotherapy-treatment-plan-optimization.sln (CLAUDE.md §5). Both
# compile the SAME src/ files, so they stay in lockstep.
#
#   cmake -S . -B build/cmake -DCMAKE_BUILD_TYPE=Release
#   cmake --build build/cmake
# ===========================================================================
cmake_minimum_required(VERSION 3.24)   # 3.24+ for first-class CUDA arch handling
project(radiotherapy-treatment-plan-optimization LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# Match the VS project's fat-binary target set (Turing/Ampere/Ada). Override on
# the command line with -DCMAKE_CUDA_ARCHITECTURES=75 for a faster local build.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75 86 89)
endif()

add_executable(radiotherapy-treatment-plan-optimization
  src/main.cu
  src/kernels.cu
  src/reference_cpu.cpp
)

# So #include "kernels.cuh" and "util/..." resolve from src/.
target_include_directories(radiotherapy-treatment-plan-optimization PRIVATE src)

# Link cuSPARSE: the two SpMVs (D x and D^T r) that dominate every FMO iteration
# are done by cuSPARSE (see src/kernels.cu). CUDA::cusparse is the imported
# target the modern CUDAToolkit package exposes (mirrors cusparse.lib in the VS
# project). CUDA::cudart is pulled in automatically by the CUDA language.
find_package(CUDAToolkit REQUIRED)
target_link_libraries(radiotherapy-treatment-plan-optimization PRIVATE CUDA::cusparse)
