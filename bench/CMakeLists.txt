add_executable(bench_sgemm bench_sgemm.cu)
target_link_libraries(bench_sgemm PRIVATE oplite CUDA::cublas)
