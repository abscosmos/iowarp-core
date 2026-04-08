#!/bin/bash
set -ex

PRESET="${IOWARP_PRESET:-cuda-delta}"

# Clean any stale build directory (preset uses ${sourceDir}/build)
rm -rf build

# Suppress GCC false positive warnings from aggressive inlining
export CXXFLAGS="${CXXFLAGS:-} -Wno-array-bounds -Wno-maybe-uninitialized -Wno-stringop-overflow"

# NOTE: CMAKE_FIND_ROOT_PATH includes /usr so CMake can find the system
# CUDA toolkit which lives outside the conda prefix on HPC clusters.
cmake --preset="${PRESET}" \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    -DCMAKE_PREFIX_PATH="${PREFIX}" \
    -DCMAKE_FIND_ROOT_PATH="${PREFIX};/usr" \
    -DWRP_CORE_ENABLE_CONDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=native

cmake --build build --parallel "${CPU_COUNT}"
cmake --install build
