// Row-wise softmax correctness test.
// Reference: safe softmax on CPU (max-subtract form).
//
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>

#include "oplite/common.cuh"
#include "oplite/softmax/softmax_online.cuh"

using namespace oplite;

namespace {

// CPU reference: safe softmax per row.
void cpu_softmax(const float* x, float* y, int M, int N) {
    for (int m = 0; m < M; ++m) {
        const float* row_x = x + m * N;
        float*       row_y = y + m * N;
        float max_v = -std::numeric_limits<float>::infinity();
        for (int n = 0; n < N; ++n) max_v = std::fmax(max_v, row_x[n]);
        float sum = 0.0f;
        for (int n = 0; n < N; ++n) sum += std::exp(row_x[n] - max_v);
        for (int n = 0; n < N; ++n) row_y[n] = std::exp(row_x[n] - max_v) / sum;
    }
}

struct Case {
    const char* name;
    int M, N;
};

bool run_one(const Case& c) {
    const size_t bytes = static_cast<size_t>(c.M) * c.N * sizeof(float);

    std::vector<float> hX(c.M * c.N), hY(c.M * c.N), hRef(c.M * c.N);
    std::mt19937 rng(0xBEEF);
    // Range includes both tails so we exercise numerical stability
    // (without max-subtract, exp would overflow).
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto& v : hX) v = dist(rng);

    float *dX, *dY;
    OPLITE_CUDA_CHECK(cudaMalloc(&dX, bytes));
    OPLITE_CUDA_CHECK(cudaMalloc(&dY, bytes));
    OPLITE_CUDA_CHECK(cudaMemcpy(dX, hX.data(), bytes, cudaMemcpyHostToDevice));

    launch_softmax_online<>(dX, dY, c.M, c.N);
    OPLITE_CUDA_KERNEL_CHECK();
    OPLITE_CUDA_CHECK(cudaDeviceSynchronize());

    OPLITE_CUDA_CHECK(cudaMemcpy(hY.data(), dY, bytes, cudaMemcpyDeviceToHost));
    cpu_softmax(hX.data(), hRef.data(), c.M, c.N);

    bool ok = allclose(hY.data(), hRef.data(), c.M * c.N,
                       /*rtol=*/1e-4f, /*atol=*/1e-5f);
    std::printf("  %-24s  M=%-5d N=%-5d  %s\n",
                c.name, c.M, c.N, ok ? "PASS" : "FAIL");

    cudaFree(dX);
    cudaFree(dY);
    return ok;
}

} // namespace

int main() {
    // Mix of sizes: small to make sure sub-BLOCK_SIZE rows work,
    // larger to cover the grid-stride loop paths.
    const Case cases[] = {
        {"tiny",     4,   64},
        {"small",   16,  256},
        {"block",   32,  512},
        {"large",   64, 4096},
        {"wide",     8, 8192},
    };

    int failed = 0;
    std::printf("[softmax_online]\n");
    for (const auto& c : cases) failed += !run_one(c);

    std::printf("\n%s (%d failures)\n",
                failed == 0 ? "ALL PASSED" : "FAILED", failed);
    return failed == 0 ? 0 : 1;
}
