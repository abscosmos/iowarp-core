#!/bin/bash
set -ex

PRESET="${IOWARP_PRESET:-release}"

# Clean any stale build directory (preset uses ${sourceDir}/build)
rm -rf build

# Suppress GCC false positive warnings from aggressive inlining
export CXXFLAGS="${CXXFLAGS:-} -Wno-array-bounds -Wno-maybe-uninitialized -Wno-stringop-overflow"

# Detect CUDA architecture: use 'native' if a GPU is present, else default
# to a portable set so the package builds on headless CI runners.
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    CUDA_ARCHS="native"
else
    CUDA_ARCHS="80-real"
fi

# On headless CI runners without an NVIDIA driver, libcuda.so is missing.
# The CUDA toolkit ships stub libraries for compile-only builds — add them
# to LIBRARY_PATH so the linker can resolve -lcuda.
if command -v nvcc &>/dev/null; then
    CUDA_STUBS="$(dirname "$(find "$(dirname "$(dirname "$(command -v nvcc)")")" -name 'libcuda.so' -path '*/stubs/*' -print -quit 2>/dev/null)" 2>/dev/null)"
    if [ -n "$CUDA_STUBS" ] && [ -d "$CUDA_STUBS" ]; then
        export LIBRARY_PATH="${CUDA_STUBS}${LIBRARY_PATH:+:$LIBRARY_PATH}"
    fi
fi

cmake --preset="${PRESET}" \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    -DCMAKE_PREFIX_PATH="${PREFIX}" \
    -DCMAKE_FIND_ROOT_PATH="${PREFIX}" \
    -DWRP_CORE_ENABLE_CONDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}"

cmake --build build --parallel "${CPU_COUNT}"
cmake --install build
