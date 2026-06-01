add_executable(test_esdf test_esdf.cpp)
target_link_libraries(test_esdf PRIVATE
  esdf_core
  GTest::gtest
  GTest::gtest_main
)
add_test(NAME test_esdf COMMAND test_esdf)
