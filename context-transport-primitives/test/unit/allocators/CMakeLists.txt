project(hermes_shm)

#------------------------------------------------------------------------------
# Build Allocator Tests
#------------------------------------------------------------------------------

add_executable(test_buddy_allocator_exec
        test_buddy_allocator.cc)
add_dependencies(test_buddy_allocator_exec hermes_shm_host)
target_link_libraries(test_buddy_allocator_exec
        hermes_shm_host Catch2::Catch2WithMain)

#------------------------------------------------------------------------------
# Test Cases
#------------------------------------------------------------------------------

add_test(NAME ctp_buddy_allocator COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_buddy_allocator_exec "[buddy_allocator]")

#------------------------------------------------------------------------------
# Install Targets
#------------------------------------------------------------------------------
install(TARGETS
        test_buddy_allocator_exec
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)

#-----------------------------------------------------------------------------
# Coverage
#-----------------------------------------------------------------------------
if(HSHM_ENABLE_COVERAGE)
    set_coverage_flags(test_buddy_allocator_exec)
endif()
