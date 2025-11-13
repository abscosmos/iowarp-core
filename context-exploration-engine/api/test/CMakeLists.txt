cmake_minimum_required(VERSION 3.20)

# Enable testing for CTest
enable_testing()

#------------------------------------------------------------------------------
# Helper function to create a unit test
#------------------------------------------------------------------------------
# Creates an executable, links dependencies, and registers with CTest
#------------------------------------------------------------------------------
function(add_cee_api_unit_test TEST_NAME TEST_SOURCE)
  # Create test executable
  add_executable(${TEST_NAME} ${TEST_SOURCE})

  # Link common dependencies
  target_link_libraries(${TEST_NAME}
    PRIVATE
      wrp_cee_api
  )

  # Install test binary
  install(TARGETS ${TEST_NAME} DESTINATION bin/tests)

  # Register with CTest
  add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})

  # Set test environment variables
  set_tests_properties(${TEST_NAME} PROPERTIES
    ENVIRONMENT "INIT_CHIMAERA=1"
    TIMEOUT 300  # 5 minute timeout
  )
endfunction()

#------------------------------------------------------------------------------
# Unit Tests
#------------------------------------------------------------------------------

# Test for context_bundle
add_cee_api_unit_test(test_context_bundle test_context_bundle.cc)

# Test for context_query
add_cee_api_unit_test(test_context_query test_context_query.cc)

# Test for context_destroy
add_cee_api_unit_test(test_context_destroy test_context_destroy.cc)

#------------------------------------------------------------------------------
# Test labels for selective execution
#------------------------------------------------------------------------------
set_tests_properties(test_context_bundle PROPERTIES LABELS "cee;api;bundle")
set_tests_properties(test_context_query PROPERTIES LABELS "cee;api;query")
set_tests_properties(test_context_destroy PROPERTIES LABELS "cee;api;destroy")
