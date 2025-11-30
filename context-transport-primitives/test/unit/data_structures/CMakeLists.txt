project(hermes_shm)

#------------------------------------------------------------------------------
# Build Data Structure Tests
#------------------------------------------------------------------------------

add_executable(test_slist_pre_exec
        test_slist_pre.cc)
add_dependencies(test_slist_pre_exec hermes_shm_host)
target_link_libraries(test_slist_pre_exec
        hermes_shm_host Catch2::Catch2WithMain)

#------------------------------------------------------------------------------
# Test Cases
#------------------------------------------------------------------------------

add_test(NAME ctp_slist_pre COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_slist_pre_exec "[slist_pre]~[atomic]")
add_test(NAME ctp_slist_pre_atomic COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_slist_pre_exec "[slist_pre][atomic]")

#------------------------------------------------------------------------------
# Install Targets
#------------------------------------------------------------------------------
install(TARGETS
        test_slist_pre_exec
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)

#-----------------------------------------------------------------------------
# Coverage
#-----------------------------------------------------------------------------
if(HSHM_ENABLE_COVERAGE)
    set_coverage_flags(test_slist_pre_exec)
endif()
