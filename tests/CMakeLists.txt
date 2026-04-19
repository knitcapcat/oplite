add_executable(test_sgemm test_sgemm.cu)
target_link_libraries(test_sgemm PRIVATE oplite CUDA::cublas)
add_test(NAME test_sgemm COMMAND test_sgemm)
