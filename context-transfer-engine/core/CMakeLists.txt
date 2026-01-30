cmake_minimum_required(VERSION 3.10)

# Compression support option
option(WRP_CORE_ENABLE_COMPRESS "Enable workflow-aware compression" OFF)

if(WRP_CORE_ENABLE_COMPRESS)
    message(STATUS "CTE Core: Compression support ENABLED")
endif()

# Create runtime library using modern ChiMod build functions
add_chimod_runtime(
  SOURCES
    src/core_runtime.cc
    src/core_config.cc
    src/core_dpe.cc
    src/autogen/core_lib_exec.cc
)

# Create client library using modern ChiMod build functions
add_chimod_client(
  SOURCES
    src/core_client.cc
    src/content_transfer_engine.cc
    src/tag.cc
)

# Add compression compile definitions and link libraries if enabled
# Note: hshm::compress includes qtable_predictor and dense_nn_predictor if available
if(WRP_CORE_ENABLE_COMPRESS)
    target_compile_definitions(wrp_cte_core_runtime PRIVATE WRP_CORE_ENABLE_COMPRESS)
    target_compile_definitions(wrp_cte_core_client PRIVATE WRP_CORE_ENABLE_COMPRESS)
    target_link_libraries(wrp_cte_core_runtime PRIVATE hshm::compress)
    target_link_libraries(wrp_cte_core_client PRIVATE hshm::compress)
endif()

# Note: Cross-namespace dependencies (chimaera admin and bdev) are automatically
# linked by add_chimod_* functions in ChimaeraCommon.cmake for non-chimaera namespaces