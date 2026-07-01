# ===========================================================================
# CMakeLists.txt  --  OPTIONAL cross-platform build (the VS .sln is required)
# ---------------------------------------------------------------------------
# Project 4.12 -- Optical Coherence Tomography Processing   (template skeleton)
#
# This is a convenience build for Linux/macOS learners and CI. On Windows the
# canonical deliverable is build/optical-coherence-tomography-processing.sln (CLAUDE.md §5). Both
# compile the SAME src/ files, so they stay in lockstep.
#
#   cmake -S . -B build/cmake -DCMAKE_BUILD_TYPE=Release
#   cmake --build build/cmake
# ===========================================================================
cmake_minimum_required(VERSION 3.24)   # 3.24+ for first-class CUDA arch handling
project(optical-coherence-tomography-processing LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# Match the VS project's fat-binary target set (Turing/Ampere/Ada). Override on
# the command line with -DCMAKE_CUDA_ARCHITECTURES=75 for a faster local build.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75 86 89)
endif()

add_executable(optical-coherence-tomography-processing
  src/main.cu
  src/kernels.cu
  src/reference_cpu.cpp
)

# So #include "kernels.cuh" and "util/..." resolve from src/.
target_include_directories(optical-coherence-tomography-processing PRIVATE src)

# Link cuFFT (batched 1D FFTs, one per A-scan). find_package supplies both the
# headers (<cufft.h>) and the import target CUDA::cufft; no manual paths needed
# (docs/BUILD_GUIDE.md §7b, docs/PATTERNS.md §5).
find_package(CUDAToolkit REQUIRED)
target_link_libraries(optical-coherence-tomography-processing PRIVATE CUDA::cufft)
