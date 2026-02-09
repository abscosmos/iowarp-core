# Benchmark directory CMakeLists.txt
cmake_minimum_required(VERSION 3.20)

# Create wrp_cte_bench executable
add_executable(wrp_cte_bench
  wrp_cte_bench.cc
)

# Link required libraries (MPI no longer required - uses worker threads instead)
target_link_libraries(wrp_cte_bench
  wrp_cte::core_client
  chimaera::cxx
)

# Set C++ standard
target_compile_features(wrp_cte_bench PRIVATE cxx_std_17)

# Install the benchmark executable
install(TARGETS wrp_cte_bench
  RUNTIME DESTINATION bin
)

# Create wrp_cte_score_bench executable (MPI-based score/demotion benchmark)
# Only build if MPI is enabled
if(WRP_CORE_ENABLE_MPI)
  add_executable(wrp_cte_score_bench
    wrp_cte_score_bench.cc
  )

  # Link required libraries including MPI
  target_link_libraries(wrp_cte_score_bench
    wrp_cte::core_client
    chimaera::cxx
    MPI::MPI_CXX
  )

  # Set C++ standard
  target_compile_features(wrp_cte_score_bench PRIVATE cxx_std_17)

  # Install the score benchmark executable
  install(TARGETS wrp_cte_score_bench
    RUNTIME DESTINATION bin
  )
else()
  message(STATUS "Skipping wrp_cte_score_bench - MPI not enabled (set WRP_CORE_ENABLE_MPI=ON)")
endif()

# NOTE: Compression benchmark (wrp_cte_compress_bench) has been moved to
# context-transfer-engine/compressor/test/
