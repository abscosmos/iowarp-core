#include <wrp_cee/api/context_interface.h>
#include <chimaera/chimaera.h>
#include <iostream>
#include <cassert>
#include <set>

/**
 * Test that context_query can be called and returns a vector
 */
void test_basic_query() {
  std::cout << "TEST: Basic query" << std::endl;

  iowarp::ContextInterface ctx_interface;

  // Query for all tags and blobs using wildcard patterns
  std::vector<std::string> results = ctx_interface.context_query(".*", ".*");

  // Result should be a vector (may be empty if no tags exist)
  // Just verify the function doesn't crash
  std::cout << "  Query returned " << results.size() << " results" << std::endl;
  std::cout << "  PASSED: Basic query test" << std::endl;
}

/**
 * Test that context_query handles specific patterns
 */
void test_specific_patterns() {
  std::cout << "TEST: Specific patterns" << std::endl;

  iowarp::ContextInterface ctx_interface;

  // Query for specific patterns
  std::vector<std::string> results1 = ctx_interface.context_query("test_.*", ".*");
  std::vector<std::string> results2 = ctx_interface.context_query(".*", "blob_[0-9]+");
  std::vector<std::string> results3 = ctx_interface.context_query("my_tag", "my_blob");

  // Just verify the function completes without crashing
  std::cout << "  Pattern 1 returned " << results1.size() << " results" << std::endl;
  std::cout << "  Pattern 2 returned " << results2.size() << " results" << std::endl;
  std::cout << "  Pattern 3 returned " << results3.size() << " results" << std::endl;
  std::cout << "  PASSED: Specific patterns test" << std::endl;
}

int main(int argc, char** argv) {
  (void)argc;  // Suppress unused parameter warning
  (void)argv;  // Suppress unused parameter warning

  std::cout << "========================================" << std::endl;
  std::cout << "ContextInterface::context_query Tests" << std::endl;
  std::cout << "========================================" << std::endl;

  try {
    test_basic_query();
    test_specific_patterns();

    std::cout << "\nAll tests PASSED!" << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "\nTest FAILED with exception: " << e.what() << std::endl;
    return 1;
  }
}
