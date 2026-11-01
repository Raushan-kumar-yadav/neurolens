add_executable(neurolens_tests
    test_main.cpp
    test_tensor.cpp
    test_ops.cpp
)
target_include_directories(neurolens_tests PRIVATE third-party)
target_link_libraries(neurolens_tests PRIVATE neurolens_core)
add_test(NAME neurolens_tests COMMAND neurolens_tests)
