// Benchmarks all SGEMM kernels in oplite against cuBLAS.
// Reports ms / GFLOPS / % of cuBLAS for a few square problem sizes.
//
// Methodology: 5 warmup iterations, then n_iter timed iterations
// wrapped in a single cudaEvent pair. Total time / n_iter.
//
#include <cublas_v2.h>
#include <cstdio>
#include <random>
#include <vector>
#include <functional>
#include <string>

#include "oplite/common.cuh"
#include "oplite/gemm/sgemm_naive.cuh"
#include "oplite/gemm/sgemm_smem.cuh"
#include "oplite/gemm/sgemm_reg_tiled.cuh"
#include "oplite/gemm/sgemm_vec.cuh"

using namespace oplite;

namespace {

constexpr int kWarmup = 5;
constexpr int kIter   = 20;

float time_kernel(std::function<void()> launch) {
    for (int i = 0; i < kWarmup; ++i) launch();
    OPLITE_CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < kIter; ++i) launch();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_total = 0.0f;
    cudaEventElapsedTime(&ms_total, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms_total / kIter;
}

double gflops(int M, int N, int K, float ms) {
    return 2.0 * static_cast<double>(M) * N * K / (ms * 1e6);
}

void cublas_sgemm_rowmajor(cublasHandle_t handle,
                           const float* dA, const float* dB, float* dC,
                           int M, int N, int K) {
    const float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha, dB, N, dA, K,
                &beta,  dC, N);
}

void run_size(int M, int N, int K, cublasHandle_t handle) {
    std::printf("\n=== M=%d N=%d K=%d ===\n", M, N, K);

    const size_t a_bytes = static_cast<size_t>(M) * K * sizeof(float);
    const size_t b_bytes = static_cast<size_t>(K) * N * sizeof(float);
    const size_t c_bytes = static_cast<size_t>(M) * N * sizeof(float);

    std::vector<float> hA(M * K), hB(K * N);
    std::mt19937 rng(0xC0FFEEu);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : hA) x = dist(rng);
    for (auto& x : hB) x = dist(rng);

    float *dA, *dB, *dC;
    OPLITE_CUDA_CHECK(cudaMalloc(&dA, a_bytes));
    OPLITE_CUDA_CHECK(cudaMalloc(&dB, b_bytes));
    OPLITE_CUDA_CHECK(cudaMalloc(&dC, c_bytes));
    OPLITE_CUDA_CHECK(cudaMemcpy(dA, hA.data(), a_bytes, cudaMemcpyHostToDevice));
    OPLITE_CUDA_CHECK(cudaMemcpy(dB, hB.data(), b_bytes, cudaMemcpyHostToDevice));

    auto bench = [&](const char* name, std::function<void()> launch,
                     float cublas_ms) {
        float ms = time_kernel(launch);
        double gf = gflops(M, N, K, ms);
        double pct = cublas_ms > 0.0f ? (cublas_ms / ms) * 100.0 : 0.0;
        std::printf("  %-18s  %8.3f ms   %9.1f GFLOPS   %6.1f%% cuBLAS\n",
                    name, ms, gf, pct);
    };

    float cublas_ms = time_kernel(
        [&]() { cublas_sgemm_rowmajor(handle, dA, dB, dC, M, N, K); });
    std::printf("  %-18s  %8.3f ms   %9.1f GFLOPS   %6.1f%% cuBLAS\n",
                "cuBLAS", cublas_ms, gflops(M, N, K, cublas_ms), 100.0);

    bench("sgemm_naive",
          [&]() { launch_sgemm_naive<>(dA, dB, dC, M, N, K); }, cublas_ms);
    bench("sgemm_smem",
          [&]() { launch_sgemm_smem<>(dA, dB, dC, M, N, K); }, cublas_ms);
    bench("sgemm_reg_tiled",
          [&]() { launch_sgemm_reg_tiled<>(dA, dB, dC, M, N, K); }, cublas_ms);
    bench("sgemm_vec",
          [&]() { launch_sgemm_vec<>(dA, dB, dC, M, N, K); }, cublas_ms);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

} // namespace

int main() {
    cublasHandle_t handle;
    cublasCreate(&handle);

    // All sizes are multiples of 128 so reg_tiled doesn't need padding.
    const int sizes[] = {1024, 2048, 4096};
    for (int s : sizes) run_size(s, s, s, handle);

    cublasDestroy(handle);
    return 0;
}
