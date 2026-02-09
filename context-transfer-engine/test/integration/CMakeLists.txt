# CMakeLists.txt for CTE Integration Tests
cmake_minimum_required(VERSION 3.10)

# Integration tests require Docker
if(WRP_CORE_ENABLE_DOCKER_CI)
    message(STATUS "CTE integration tests enabled (Docker CI)")
    add_subdirectory(distributed)
else()
    message(STATUS "CTE integration tests disabled (set WRP_CORE_ENABLE_DOCKER_CI=ON to enable)")
endif()
