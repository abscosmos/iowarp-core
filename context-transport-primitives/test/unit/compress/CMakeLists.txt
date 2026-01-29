project(hermes_shm)

#------------------------------------------------------------------------------
# Build Tests
#------------------------------------------------------------------------------

# Basic compression unit tests
add_executable(test_compress_exec
        ${TEST_MAIN}/main.cc
        test_init.cc
        test_compress.cc)

# Unified compression benchmark (main benchmark executable)
add_executable(benchmark_compression_unified_exec
        ${TEST_MAIN}/main.cc
        test_init.cc
        test_compression_benchmark.cc)

# Q-Table predictor test
add_executable(test_qtable_predictor
        test_qtable_predictor.cc)

# Compression contention benchmark (motivation section)
add_executable(benchmark_compression_contention_exec
        test_compression_contention.cc)

#------------------------------------------------------------------------------
# Link Dependencies
#------------------------------------------------------------------------------

add_dependencies(test_compress_exec hermes_shm_host)
target_link_libraries(test_compress_exec
        hermes_shm_host hshm::compress Catch2::Catch2)
target_compile_definitions(test_compress_exec
        PRIVATE HSHM_ENABLE_COMPRESS=1)

add_dependencies(benchmark_compression_unified_exec hermes_shm_host)

# Link lossy compression libraries directly
find_library(ZFP_LIBRARY NAMES zfp PATHS /usr/local/lib)
find_library(SZ3_LIBRARY NAMES SZ3c PATHS /usr/local/lib)
find_library(SZ_LIBRARY NAMES SZ PATHS /usr/local/lib)
find_library(FPZIP_LIBRARY NAMES fpzip PATHS /usr/local/lib)

set(LOSSY_LIBS "")
if(ZFP_LIBRARY)
    list(APPEND LOSSY_LIBS ${ZFP_LIBRARY})
    message(STATUS "Found ZFP library: ${ZFP_LIBRARY}")
endif()
if(SZ3_LIBRARY)
    list(APPEND LOSSY_LIBS ${SZ3_LIBRARY})
    message(STATUS "Found SZ3 library: ${SZ3_LIBRARY}")
endif()
if(SZ_LIBRARY)
    list(APPEND LOSSY_LIBS ${SZ_LIBRARY})
    message(STATUS "Found SZ library: ${SZ_LIBRARY}")
endif()
if(FPZIP_LIBRARY)
    list(APPEND LOSSY_LIBS ${FPZIP_LIBRARY})
    message(STATUS "Found FPZIP library: ${FPZIP_LIBRARY}")
endif()

target_link_libraries(benchmark_compression_unified_exec
        hermes_shm_host hshm::compress Catch2::Catch2 ${LOSSY_LIBS})
target_compile_definitions(benchmark_compression_unified_exec
        PRIVATE HSHM_ENABLE_COMPRESS=1)

# Link Q-Table predictor test (requires nlohmann_json)
find_package(nlohmann_json QUIET)
if(nlohmann_json_FOUND AND TARGET qtable_predictor)
    target_link_libraries(test_qtable_predictor
            qtable_predictor)
    set_property(TARGET test_qtable_predictor PROPERTY CXX_STANDARD 17)
    message(STATUS "Q-Table predictor test enabled")
else()
    message(STATUS "Q-Table predictor not available (requires nlohmann_json)")
    set_target_properties(test_qtable_predictor PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()

# Link compression contention benchmark
add_dependencies(benchmark_compression_contention_exec hermes_shm_host)
target_link_libraries(benchmark_compression_contention_exec
        hermes_shm_host hshm::compress pthread)
target_compile_definitions(benchmark_compression_contention_exec
        PRIVATE HSHM_ENABLE_COMPRESS=1)

#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

add_test(NAME ctp_compress COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_compress_exec "~[error=FatalError]")

#------------------------------------------------------------------------------
# Install Targets
#------------------------------------------------------------------------------

install(TARGETS
        test_compress_exec
        benchmark_compression_unified_exec
        benchmark_compression_contention_exec
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)

#------------------------------------------------------------------------------
# Coverage
#------------------------------------------------------------------------------

if(HSHM_ENABLE_COVERAGE)
    set_coverage_flags(test_compress_exec)
endif()
