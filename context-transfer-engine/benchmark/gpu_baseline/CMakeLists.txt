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

# ============================================================
# Benchmark 2: Remote HBM via NVSHMEM (requires WRP_CORE_ENABLE_NVSHMEM=ON)
# ============================================================

if(WRP_CORE_ENABLE_NVSHMEM)
  # NVSHMEM_HOME env var or -DNVSHMEM_HOME=/path sets the search prefix.
  set(_NVSHMEM_PREFIX "$ENV{NVSHMEM_HOME}" CACHE PATH "NVSHMEM install prefix")
  if(NOT _NVSHMEM_PREFIX AND DEFINED ENV{NVSHMEM_HOME})
    set(_NVSHMEM_PREFIX "$ENV{NVSHMEM_HOME}")
  endif()

  find_path(NVSHMEM_INCLUDE_DIR nvshmem.h
    HINTS ${_NVSHMEM_PREFIX}/include
    PATHS /usr/local/nvshmem/include /opt/nvshmem/include)
  # NVSHMEM 3.x splits into nvshmem_host (shared) + nvshmem_device (static).
  # Older versions have a single libnvshmem. Try both naming conventions.
  find_library(NVSHMEM_HOST_LIBRARY nvshmem_host
    HINTS ${_NVSHMEM_PREFIX}/lib
    PATHS /usr/local/nvshmem/lib /opt/nvshmem/lib)
  find_library(NVSHMEM_DEVICE_LIBRARY nvshmem_device
    HINTS ${_NVSHMEM_PREFIX}/lib
    PATHS /usr/local/nvshmem/lib /opt/nvshmem/lib)
  # Fallback: single libnvshmem (NVSHMEM 2.x layout)
  find_library(NVSHMEM_LIBRARY nvshmem
    HINTS ${_NVSHMEM_PREFIX}/lib
    PATHS /usr/local/nvshmem/lib /opt/nvshmem/lib)

  if(NVSHMEM_INCLUDE_DIR AND (NVSHMEM_HOST_LIBRARY OR NVSHMEM_LIBRARY))
    set(HAVE_NVSHMEM TRUE)
    if(NVSHMEM_HOST_LIBRARY)
      message(STATUS "gpu_baseline: nvshmem found (3.x layout) — host: ${NVSHMEM_HOST_LIBRARY}")
      if(NVSHMEM_DEVICE_LIBRARY)
        message(STATUS "gpu_baseline: nvshmem device lib: ${NVSHMEM_DEVICE_LIBRARY}")
      endif()
    else()
      message(STATUS "gpu_baseline: nvshmem found at ${NVSHMEM_LIBRARY}")
    endif()
  else()
    set(HAVE_NVSHMEM FALSE)
    message(STATUS "gpu_baseline: nvshmem not found — skipping bench_remote_hbm (set NVSHMEM_HOME or install nvshmem)")
  endif()
else()
  set(HAVE_NVSHMEM FALSE)
  message(STATUS "gpu_baseline: WRP_CORE_ENABLE_NVSHMEM=OFF — skipping bench_remote_hbm")
endif()

if(HAVE_NVSHMEM)
  # MPI is required for the standard nvshmem multi-PE launch model
  find_package(MPI REQUIRED COMPONENTS CXX)

  # Build link list: prefer 3.x split libs, fall back to single libnvshmem
  set(_REMOTE_HBM_LIBS CUDA::cudart)
  if(NVSHMEM_HOST_LIBRARY)
    list(APPEND _REMOTE_HBM_LIBS ${NVSHMEM_HOST_LIBRARY})
  endif()
  if(NVSHMEM_DEVICE_LIBRARY)
    list(APPEND _REMOTE_HBM_LIBS ${NVSHMEM_DEVICE_LIBRARY})
  elseif(NVSHMEM_LIBRARY)
    list(APPEND _REMOTE_HBM_LIBS ${NVSHMEM_LIBRARY})
  endif()
  list(APPEND _REMOTE_HBM_LIBS MPI::MPI_CXX)

  add_cuda_executable(gpu_baseline_remote_hbm TRUE
    bench_remote_hbm.cu
    INCLUDE_DIRS
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${NVSHMEM_INCLUDE_DIR}
    LINK_LIBS
      ${_REMOTE_HBM_LIBS}
    DEFS
      HAVE_NVSHMEM=1
  )

  # NVSHMEM device code requires relocatable device code for NVCC
  if(NOT WRP_CORE_CLANG_CUDA_COMPILER)
    set_target_properties(gpu_baseline_remote_hbm PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON)
  endif()

  install(TARGETS gpu_baseline_remote_hbm RUNTIME DESTINATION bin)

  # Sanity ctest (loopback: PE 0 puts to PE 0, single warp, 4 KB)
  # Requires nvshmrun or mpirun in PATH
  add_test(NAME gpu_baseline_remote_hbm_sanity
    COMMAND nvshmrun -n 1 $<TARGET_FILE:gpu_baseline_remote_hbm>
      --blocks 1 --threads 32
      --bytes-per-warp 4096
      --iterations 10
      --mode put)
  set_tests_properties(gpu_baseline_remote_hbm_sanity PROPERTIES
    TIMEOUT 60 LABELS "gpu;nvshmem;msan_skip")

  # 2-PE test (requires 2 GPUs — will fail gracefully on 1-GPU machines)
  add_test(NAME gpu_baseline_remote_hbm_2pe
    COMMAND nvshmrun -n 2 $<TARGET_FILE:gpu_baseline_remote_hbm>
      --blocks 1 --threads 32
      --bytes-per-warp 4096
      --iterations 10
      --mode put)
  set_tests_properties(gpu_baseline_remote_hbm_2pe PROPERTIES
    TIMEOUT 60 LABELS "gpu;nvshmem;multi_gpu;msan_skip")
endif()
