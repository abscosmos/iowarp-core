cmake_minimum_required(VERSION 3.20)
project(wrp_cee_api VERSION 1.0.0)

#------------------------------------------------------------------------------
# Find Required Packages
#------------------------------------------------------------------------------
# Skip find_package if targets already exist (unified build as subdirectory)
if(NOT TARGET chimaera::cxx AND NOT TARGET chimaera_cxx)
  find_package(chimaera REQUIRED)
  list(APPEND CMAKE_MODULE_PATH "${chimaera_DIR}")
  include(ChimaeraCommon)
  message(STATUS "Found Chimaera package")
else()
  message(STATUS "Using Chimaera as subdirectory")
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../context-runtime/cmake")
  include(ChimaeraCommon)
endif()

# Find wrp_cte_core package
if(NOT TARGET wrp_cte::core_client AND NOT TARGET wrp_cte_core_client)
  find_package(wrp_cte_core REQUIRED)
  message(STATUS "Found wrp_cte_core package")
else()
  message(STATUS "Using wrp_cte_core as subdirectory")
endif()

# Find wrp_cae_core package
if(NOT TARGET wrp_cae::core_client AND NOT TARGET wrp_cae_core_client)
  find_package(wrp_cae_core REQUIRED)
  message(STATUS "Found wrp_cae_core package")
else()
  message(STATUS "Using wrp_cae_core as subdirectory")
endif()

#------------------------------------------------------------------------------
# Context Interface Library
#------------------------------------------------------------------------------
add_library(wrp_cee_api SHARED
  src/context_interface.cc
)

add_library(wrp_cee::api ALIAS wrp_cee_api)

target_include_directories(wrp_cee_api
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(wrp_cee_api
  PUBLIC
    chimaera::cxx
    wrp_cte_core_client
    wrp_cae_core_client
)

# Set library properties
set_target_properties(wrp_cee_api PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION 1
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

#------------------------------------------------------------------------------
# Python Bindings (using nanobind)
#------------------------------------------------------------------------------
# Find nanobind
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(nanobind CONFIG)

if(nanobind_FOUND)
  message(STATUS "nanobind found - building Python bindings")

  # Create Python module
  nanobind_add_module(iowarp_cee_api
    src/python_bindings.cc
  )

  target_link_libraries(iowarp_cee_api
    PRIVATE
      wrp_cee_api
  )

  # Install Python module
  install(TARGETS iowarp_cee_api
    LIBRARY DESTINATION ${Python_SITEARCH}
  )
else()
  message(WARNING "nanobind not found - skipping Python bindings")
endif()

#------------------------------------------------------------------------------
# Installation
#------------------------------------------------------------------------------
install(TARGETS wrp_cee_api
  EXPORT wrp_cee_api_targets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

install(DIRECTORY include/
  DESTINATION include
  FILES_MATCHING PATTERN "*.h"
)

install(EXPORT wrp_cee_api_targets
  FILE wrp_cee_api_targets.cmake
  NAMESPACE wrp_cee::
  DESTINATION lib/cmake/wrp_cee_api
)

# Create and install package config files
include(CMakePackageConfigHelpers)

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/wrp_cee_api-config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/wrp_cee_api-config.cmake"
  INSTALL_DESTINATION lib/cmake/wrp_cee_api
)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/wrp_cee_api-config-version.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/wrp_cee_api-config.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/wrp_cee_api-config-version.cmake"
  DESTINATION lib/cmake/wrp_cee_api
)

#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------
if(BUILD_TESTING)
  add_subdirectory(test)
endif()
