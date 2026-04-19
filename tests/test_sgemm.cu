// Numerical correctness for every SGEMM in oplite.
//
// We use cuBLAS as the oracle. cuBLAS is column-major; our kernels are
// row-major. The standard trick to call column-major cuBLAS on
// row-major inputs is to compute  C^T = B^T * A^T, which in cuBLAS-speak
// is just a no-transpose call with the operand order swapped:
//
//   C(M,N) = A(M,K) * B(K,N)            (row-major view)
//   <==> C^T(N,M) = B^T(N,K) * A^T(K,M) (column-major view)
//   <==> cublasSgemm(N, M, K, B, N, A, K, C, N) with no transposes
//
#include <cublas_v2.h>
#include <cstdio>
#include <random>
#include <vector>
#include <string>

#include "oplite/common.cuh"
#include "oplite/gemm/sgemm_naive.cuh"
#include "oplite/gemm/sgemm_smem.cuh"
#include "oplite/gemm/sgemm_reg_tiled.cuh"
#include "oplite/gemm/sgemm_vec.cuh"

using namespace oplite;

namespace {

void cublas_sgemm_rowmajor(cublasHandle_t handle,
                           const float* dA, const float* dB, float* dC,
                           int M, int N, int K) {
    const float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                dB, N,
                dA, K,
                &beta,
                dC, N);
}

struct Case {
    const char* name;
    int M, N, K;
};

template <typename Launch>
bool run_one(const char* kernel_name, const Case& c, Launch&& launch,
             cublasHandle_t handle) {
    const size_t a_bytes = static_cast<size_t>(c.M) * c.K * sizeof(float);
    const size_t b_bytes = static_cast<size_t>(c.K) * c.N * sizeof(float);
    const size_t c_bytes = static_cast<size_t>(c.M) * c.N * sizeof(float);

    std::vector<float> hA(c.M * c.K), hB(c.K * c.N);
    std::mt19937 rng(0xC0FFEEu);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : hA) x = dist(rng);
    for (auto& x : hB) x = dist(rng);

    float *dA, *dB, *dC, *dRef;
    OPLITE_CUDA_CHECK(cudaMalloc(&dA,   a_bytes));
    OPLITE_CUDA_CHECK(cudaMalloc(&dB,   b_bytes));
    OPLITE_CUDA_CHECK(cudaMalloc(&dC,   c_bytes));
    OPLITE_CUDA_CHECK(cudaMalloc(&dRef, c_bytes));
    OPLITE_CUDA_CHECK(cudaMemcpy(dA, hA.data(), a_bytes, cudaMemcpyHostToDevice));
    OPLITE_CUDA_CHECK(cudaMemcpy(dB, hB.data(), b_bytes, cudaMemcpyHostToDevice));

    cublas_sgemm_rowmajor(handle, dA, dB, dRef, c.M, c.N, c.K);
    launch(dA, dB, dC, c.M, c.N, c.K);
    OPLITE_CUDA_KERNEL_CHECK();
    OPLITE_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> hC(c.M * c.N), hRef(c.M * c.N);
    OPLITE_CUDA_CHECK(cudaMemcpy(hC.data(),   dC,   c_bytes, cudaMemcpyDeviceToHost));
    OPLITE_CUDA_CHECK(cudaMemcpy(hRef.data(), dRef, c_bytes, cudaMemcpyDeviceToHost));

    bool ok = allclose(hC.data(), hRef.data(), c.M * c.N,
                       /*rtol=*/1e-3f, /*atol=*/1e-3f);
    std::printf("  %-22s  M=%-5d N=%-5d K=%-5d  %s\n",
                kernel_name, c.M, c.N, c.K, ok ? "PASS" : "FAIL");

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFree(dRef);
    return ok;
}

} // namespace

int main() {
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Sizes chosen so reg_tiled (BM=BN=128) divides cleanly.
    const Case cases[] = {
        {"small",  128,  128,  128},
        {"medium", 512,  512,  512},
        {"large",  1024, 1024, 1024},
    };

    int failed = 0;
    for (const auto& c : cases) {
        std::printf("[%s]\n", c.name);
        failed += !run_one("sgemm_naive", c,
            [](auto... args){ launch_sgemm_naive<>(args...); }, handle);
        failed += !run_one("sgemm_smem", c,
            [](auto... args){ launch_sgemm_smem<>(args...); }, handle);
        failed += !run_one("sgemm_reg_tiled", c,
            [](auto... args){ launch_sgemm_reg_tiled<>(args...); }, handle);
        failed += !run_one("sgemm_vec", c,
            [](auto... args){ launch_sgemm_vec<>(args...); }, handle);
    }

    cublasDestroy(handle);
    std::printf("\n%s (%d failures)\n", failed == 0 ? "ALL PASSED" : "FAILED", failed);
    return failed == 0 ? 0 : 1;
}
