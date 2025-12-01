project(hermes_shm)

#------------------------------------------------------------------------------
# Build Allocator Tests
#------------------------------------------------------------------------------

add_executable(test_alloc_exec
        test_alloc.cc)
add_dependencies(test_alloc_exec hermes_shm_host)
target_link_libraries(test_alloc_exec
        hermes_shm_host Catch2::Catch2WithMain)

add_executable(test_sub_alloc_exec
        test_sub_alloc.cc)
add_dependencies(test_sub_alloc_exec hermes_shm_host)
target_link_libraries(test_sub_alloc_exec
        hermes_shm_host Catch2::Catch2WithMain)

add_executable(test_arena_alloc_exec
        test_arena_alloc.cc)
add_dependencies(test_arena_alloc_exec hermes_shm_host)
target_link_libraries(test_arena_alloc_exec
        hermes_shm_host Catch2::Catch2WithMain)

add_executable(test_buddy_allocator_exec
        test_buddy_allocator.cc)
add_dependencies(test_buddy_allocator_exec hermes_shm_host)
target_link_libraries(test_buddy_allocator_exec
        hermes_shm_host Catch2::Catch2WithMain)

add_executable(test_mp_allocator_exec
        test_mp_allocator.cc)
add_dependencies(test_mp_allocator_exec hermes_shm_host)
target_link_libraries(test_mp_allocator_exec
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

add_test(NAME ctp_sub_alloc COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_sub_alloc_exec "[SubAllocator]~[nested]")
add_test(NAME ctp_sub_alloc_nested COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_sub_alloc_exec "[SubAllocator][nested]")

add_test(NAME ctp_arena_alloc COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_arena_alloc_exec "[ArenaAllocator]~[atomic]")
add_test(NAME ctp_arena_alloc_atomic COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_arena_alloc_exec "[ArenaAllocator][atomic]")

add_test(NAME ctp_buddy_allocator COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_buddy_allocator_exec "[buddy_allocator]")

add_test(NAME ctp_mp_allocator COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_mp_allocator_exec "[mp_allocator]")

#------------------------------------------------------------------------------
# Install Targets
#------------------------------------------------------------------------------
install(TARGETS
        test_alloc_exec
        test_sub_alloc_exec
        test_arena_alloc_exec
        test_buddy_allocator_exec
        test_mp_allocator_exec
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)

#-----------------------------------------------------------------------------
# Coverage
#-----------------------------------------------------------------------------
if(HSHM_ENABLE_COVERAGE)
    set_coverage_flags(test_alloc_exec)
    set_coverage_flags(test_sub_alloc_exec)
    set_coverage_flags(test_arena_alloc_exec)
    set_coverage_flags(test_buddy_allocator_exec)
    set_coverage_flags(test_mp_allocator_exec)
endif()
