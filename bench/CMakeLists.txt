add_executable(bench_sgemm bench_sgemm.cu)
target_link_libraries(bench_sgemm PRIVATE oplite CUDA::cublas)

add_executable(bench_attention bench_attention.cu)
target_link_libraries(bench_attention PRIVATE oplite)
