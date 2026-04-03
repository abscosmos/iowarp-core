cmake_minimum_required(VERSION 3.20)

# gpu_baseline — standalone GPU transport microbenchmarks
# No Chimaera / HermesShm dependency; only CUDA::cudart (+ optional libnuma).
# Included by context-transfer-engine/benchmark/CMakeLists.txt inside the
# if(WRP_CORE_ENABLE_CUDA) block.

# CUDAToolkit provides CUDA::cudart as an imported CMake target with the
# correct include directories.  Call find_package here so that the target is
# guaranteed to exist even when this subdirectory is processed before any
# other CMakeLists that does the same find_package call.
find_package(CUDAToolkit REQUIRED)

# --- Optional libnuma detection ---
find_library(NUMA_LIBRARY numa)
find_path(NUMA_INCLUDE_DIR numa.h)
if(NUMA_LIBRARY AND NUMA_INCLUDE_DIR)
  set(HAVE_LIBNUMA TRUE)
  message(STATUS "gpu_baseline: found libnuma at ${NUMA_LIBRARY}")
else()
  set(HAVE_LIBNUMA FALSE)
  message(STATUS "gpu_baseline: libnuma not found — NUMA pinning disabled (no-op on single-NUMA machines)")
endif()

# --- Benchmark 1: Pinned host memory bandwidth/latency ---
# Build up optional args so libnuma defs/includes reach the Clang CUDA command.
set(_PINNED_HOST_DEFS "")
set(_PINNED_HOST_INCDIRS "${CMAKE_CURRENT_SOURCE_DIR}")
set(_PINNED_HOST_LIBS CUDA::cudart)
if(HAVE_LIBNUMA)
  list(APPEND _PINNED_HOST_DEFS HAVE_LIBNUMA=1)
  list(APPEND _PINNED_HOST_INCDIRS ${NUMA_INCLUDE_DIR})
  list(APPEND _PINNED_HOST_LIBS ${NUMA_LIBRARY})
endif()

add_cuda_executable(gpu_baseline_pinned_host TRUE
    bench_pinned_host.cu
    INCLUDE_DIRS
        ${_PINNED_HOST_INCDIRS}
    LINK_LIBS
        ${_PINNED_HOST_LIBS}
    DEFS
        ${_PINNED_HOST_DEFS}
)

install(TARGETS gpu_baseline_pinned_host RUNTIME DESTINATION bin)

# --- Sanity ctest: 1 warp, 4 KB, 10 iterations — should finish in < 30 s ---
add_test(NAME gpu_baseline_pinned_host_sanity
  COMMAND gpu_baseline_pinned_host
    --blocks 1 --threads 32
    --bytes-per-warp 4096
    --iterations 10
    --gpu-id 0 --numa-node 0)
set_tests_properties(gpu_baseline_pinned_host_sanity PROPERTIES
  TIMEOUT 30 LABELS "gpu;msan_skip")

# --- Validation ctest: 1 warp, 4 KB, 1 iteration + readback check ---
add_test(NAME gpu_baseline_pinned_host_validate
  COMMAND gpu_baseline_pinned_host
    --blocks 1 --threads 32
    --bytes-per-warp 4096
    --iterations 1
    --gpu-id 0 --numa-node 0
    --validate)
set_tests_properties(gpu_baseline_pinned_host_validate PROPERTIES
  TIMEOUT 30 LABELS "gpu;msan_skip")
