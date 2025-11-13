cmake_minimum_required(VERSION 3.20)
project(context_exploration_engine)

# Set root directory for this component
set(WRP_CEE_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

# Add api subdirectory for ContextInterface C++ wrapper and Python bindings
message(STATUS "Context Exploration Engine: Building API components")
add_subdirectory(api)

# hdf-compass and mcp-hdf-demo are standalone components
message(STATUS "  hdf-compass and mcp-hdf-demo are standalone components")
message(STATUS "  To build them, navigate to their respective directories")
