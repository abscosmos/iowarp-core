#include <wrp_cee/api/context_interface.h>
#include <chimaera/chimaera.h>
#include <iostream>
#include <cassert>

/**
 * Test that context_destroy can handle empty context list
 */
void test_empty_context_list() {
  std::cout << "TEST: Empty context list" << std::endl;

  iowarp::ContextInterface ctx_interface;
  std::vector<std::string> empty_list;

  // Empty list should return success (0)
  int result = ctx_interface.context_destroy(empty_list);
  assert(result == 0 && "Empty context list should return success");

  std::cout << "  PASSED: Empty context list test" << std::endl;
}

/**
 * Test that context_destroy handles non-existent contexts gracefully
 */
void test_nonexistent_context() {
  std::cout << "TEST: Non-existent context" << std::endl;

  iowarp::ContextInterface ctx_interface;
  std::vector<std::string> contexts;
  contexts.push_back("definitely_does_not_exist_context_12345");

  // Non-existent context should be handled gracefully
  int result = ctx_interface.context_destroy(contexts);

  // Result could be 0 or non-zero depending on CTE behavior
  // Just verify the function completes without crashing
  std::cout << "  Destroy returned code: " << result << std::endl;
  std::cout << "  PASSED: Non-existent context test" << std::endl;
}

/**
 * Test that context_destroy handles special characters
 */
void test_special_characters() {
  std::cout << "TEST: Special characters" << std::endl;

  iowarp::ContextInterface ctx_interface;
  std::vector<std::string> contexts;
  contexts.push_back("test-context_with.special:chars");

  int result = ctx_interface.context_destroy(contexts);

  // Should handle special characters without crashing
  std::cout << "  Destroy returned code: " << result << std::endl;
  std::cout << "  PASSED: Special characters test" << std::endl;
}

int main(int argc, char** argv) {
  (void)argc;  // Suppress unused parameter warning
  (void)argv;  // Suppress unused parameter warning

  std::cout << "========================================" << std::endl;
  std::cout << "ContextInterface::context_destroy Tests" << std::endl;
  std::cout << "========================================" << std::endl;

  try {
    test_empty_context_list();
    test_nonexistent_context();
    test_special_characters();

    std::cout << "\nAll tests PASSED!" << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "\nTest FAILED with exception: " << e.what() << std::endl;
    return 1;
  }
}
