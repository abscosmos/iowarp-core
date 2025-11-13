#include <wrp_cee/api/context_interface.h>
#include <wrp_cae/core/factory/assimilation_ctx.h>
#include <chimaera/chimaera.h>
#include <iostream>
#include <cassert>

/**
 * Test that context_bundle can be created and handles empty bundles
 */
void test_empty_bundle() {
  std::cout << "TEST: Empty bundle" << std::endl;

  iowarp::ContextInterface ctx_interface;
  std::vector<wrp_cae::core::AssimilationCtx> empty_bundle;

  // Empty bundle should return success (0)
  int result = ctx_interface.context_bundle(empty_bundle);
  assert(result == 0 && "Empty bundle should return success");

  std::cout << "  PASSED: Empty bundle test" << std::endl;
}

/**
 * Test AssimilationCtx constructor with all parameters
 */
void test_assimilation_ctx_constructor() {
  std::cout << "TEST: AssimilationCtx constructor" << std::endl;

  wrp_cae::core::AssimilationCtx ctx(
      "file::/path/to/source.dat",
      "iowarp::dest_tag",
      "binary",
      "dependency_id",
      1024,
      2048,
      "src_access_token",
      "dst_access_token");

  assert(ctx.src == "file::/path/to/source.dat");
  assert(ctx.dst == "iowarp::dest_tag");
  assert(ctx.format == "binary");
  assert(ctx.depends_on == "dependency_id");
  assert(ctx.range_off == 1024);
  assert(ctx.range_size == 2048);
  assert(ctx.src_token == "src_access_token");
  assert(ctx.dst_token == "dst_access_token");

  std::cout << "  PASSED: AssimilationCtx constructor test" << std::endl;
}

int main(int argc, char** argv) {
  (void)argc;  // Suppress unused parameter warning
  (void)argv;  // Suppress unused parameter warning

  std::cout << "========================================" << std::endl;
  std::cout << "ContextInterface::context_bundle Tests" << std::endl;
  std::cout << "========================================" << std::endl;

  try {
    test_empty_bundle();
    test_assimilation_ctx_constructor();

    std::cout << "\nAll tests PASSED!" << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "\nTest FAILED with exception: " << e.what() << std::endl;
    return 1;
  }
}
