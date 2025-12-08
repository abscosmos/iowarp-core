project(hermes_shm)

#------------------------------------------------------------------------------
# Build Data Structure Tests
#------------------------------------------------------------------------------

add_executable(test_slist_pre_exec
        test_slist_pre.cc)
add_dependencies(test_slist_pre_exec hermes_shm_host)
target_link_libraries(test_slist_pre_exec
        hermes_shm_host Catch2::Catch2WithMain)

add_executable(test_rb_tree_pre_exec
        test_rb_tree_pre.cc)
add_dependencies(test_rb_tree_pre_exec hermes_shm_host)
target_link_libraries(test_rb_tree_pre_exec
        hermes_shm_host Catch2::Catch2WithMain)

add_executable(test_vector_exec
        vector.cc)
add_dependencies(test_vector_exec hermes_shm_host)
target_link_libraries(test_vector_exec
        hermes_shm_host)

#------------------------------------------------------------------------------
# Test Cases
#------------------------------------------------------------------------------

add_test(NAME ctp_slist_pre COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_slist_pre_exec "[slist_pre]~[atomic]")
add_test(NAME ctp_slist_pre_atomic COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_slist_pre_exec "[slist_pre][atomic]")

add_test(NAME ctp_rb_tree_pre COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_rb_tree_pre_exec "[rb_tree_pre]~[atomic]")
add_test(NAME ctp_rb_tree_pre_atomic COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_rb_tree_pre_exec "[rb_tree_pre][atomic]")

add_test(NAME ctp_vector COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_vector_exec "[vector]")
add_test(NAME ctp_vector_pod COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_vector_exec "[vector][pod]")
add_test(NAME ctp_vector_non_pod COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_vector_exec "[vector][non-pod]")
add_test(NAME ctp_vector_edge COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_vector_exec "[vector][edge]")
add_test(NAME ctp_vector_stress COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_vector_exec "[vector][stress]")

#------------------------------------------------------------------------------
# Install Targets
#------------------------------------------------------------------------------
install(TARGETS
        test_slist_pre_exec
        test_rb_tree_pre_exec
        test_vector_exec
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)

#-----------------------------------------------------------------------------
# Coverage
#-----------------------------------------------------------------------------
if(HSHM_ENABLE_COVERAGE)
    set_coverage_flags(test_slist_pre_exec)
    set_coverage_flags(test_rb_tree_pre_exec)
    set_coverage_flags(test_vector_exec)
endif()
