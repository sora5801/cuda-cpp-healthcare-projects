# ===========================================================================
# CMakeLists.txt  --  OPTIONAL cross-platform build (the VS .sln is required)
# ---------------------------------------------------------------------------
# Project 2.8 -- GPU Molecular Visualization & Ray Tracing   (template skeleton)
#
# This is a convenience build for Linux/macOS learners and CI. On Windows the
# canonical deliverable is build/gpu-molecular-visualization-ray-tracing.sln (CLAUDE.md §5). Both
# compile the SAME src/ files, so they stay in lockstep.
#
#   cmake -S . -B build/cmake -DCMAKE_BUILD_TYPE=Release
#   cmake --build build/cmake
# ===========================================================================
cmake_minimum_required(VERSION 3.24)   # 3.24+ for first-class CUDA arch handling
project(gpu-molecular-visualization-ray-tracing LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# Match the VS project's fat-binary target set (Turing/Ampere/Ada). Override on
# the command line with -DCMAKE_CUDA_ARCHITECTURES=75 for a faster local build.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75 86 89)
endif()

add_executable(gpu-molecular-visualization-ray-tracing
  src/main.cu
  src/kernels.cu
  src/reference_cpu.cpp
)

# So #include "kernels.cuh" and "util/..." resolve from src/.
target_include_directories(gpu-molecular-visualization-ray-tracing PRIVATE src)

# --fmad=false : disable fused-multiply-add CONTRACTION on the device so the GPU
# computes a*b+c as a separate multiply then add, like the host C++ compiler.
# This MINIMIZES CPU-vs-GPU divergence in the render (it does not make them
# bit-identical -- the host/device cosf/sinf/sqrtf still differ in the last bit,
# flipping a few silhouette-edge pixels). Keep this in lockstep with the <FMAD>
# setting in build/gpu-molecular-visualization-ray-tracing.vcxproj. See THEORY.md
# "How we verify correctness" and docs/PATTERNS.md §4.
target_compile_options(gpu-molecular-visualization-ray-tracing PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
