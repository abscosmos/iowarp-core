project(hermes_shm)

#------------------------------------------------------------------------------
# Build Allocator Tests
#------------------------------------------------------------------------------

add_executable(test_alloc_exec
        test_alloc.cc)
add_dependencies(test_alloc_exec hermes_shm_host)
target_link_libraries(test_alloc_exec
        hermes_shm_host Catch2::Catch2WithMain)

#------------------------------------------------------------------------------
# Test Cases
#------------------------------------------------------------------------------

add_test(NAME ctp_alloc_immediate COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_alloc_exec "[MallocAllocator]~[multithread]~[all]")
add_test(NAME ctp_alloc_multithread COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_alloc_exec "[MallocAllocator][multithread]")
add_test(NAME ctp_alloc_all COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_alloc_exec "[MallocAllocator][all]")

#------------------------------------------------------------------------------
# Install Targets
#------------------------------------------------------------------------------
install(TARGETS
        test_alloc_exec
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)

#-----------------------------------------------------------------------------
# Coverage
#-----------------------------------------------------------------------------
if(HSHM_ENABLE_COVERAGE)
    set_coverage_flags(test_alloc_exec)
endif()
