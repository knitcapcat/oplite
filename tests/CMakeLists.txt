add_executable(test_sgemm test_sgemm.cu)
target_link_libraries(test_sgemm PRIVATE oplite CUDA::cublas)
add_test(NAME test_sgemm COMMAND test_sgemm)

add_executable(test_softmax test_softmax.cu)
target_link_libraries(test_softmax PRIVATE oplite)
add_test(NAME test_softmax COMMAND test_softmax)

add_executable(test_attention test_attention.cu)
target_link_libraries(test_attention PRIVATE oplite)
add_test(NAME test_attention COMMAND test_attention)
