// Correctness test for unfused attention forward.
// Reference: full FP32 attention on CPU (O(N^2 * D)).
//
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>

#include "oplite/common.cuh"
#include "oplite/attention/unfused_forward.cuh"

using namespace oplite;

namespace {

// CPU reference: S = softmax(scale * Q @ K^T),  O = S @ V.
void cpu_attention_forward(
    const float* Q, const float* K, const float* V, float* O,
    int N, int D, float scale)
{
    std::vector<float> S(static_cast<size_t>(N) * N);

    // S[i,j] = scale * <Q[i,:], K[j,:]>
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < D; ++k) {
                sum += Q[i * D + k] * K[j * D + k];
            }
            S[i * N + j] = sum * scale;
        }
    }

    // Row softmax (safe, max-subtract form).
    for (int i = 0; i < N; ++i) {
        float m = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < N; ++j) m = std::fmax(m, S[i * N + j]);
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) sum += std::exp(S[i * N + j] - m);
        for (int j = 0; j < N; ++j) {
            S[i * N + j] = std::exp(S[i * N + j] - m) / sum;
        }
    }

    // O = S @ V       (N, D) = (N, N) @ (N, D)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < D; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += S[i * N + k] * V[k * D + j];
            }
            O[i * D + j] = sum;
        }
    }
}

struct Case {
    const char* name;
    int N, D;
};

bool run_one(const Case& c) {
    const size_t qkv_bytes = static_cast<size_t>(c.N) * c.D * sizeof(float);
    const size_t ws_bytes  = unfused_attention_forward_workspace_bytes(c.N, c.D);
    const float  scale     = 1.0f / std::sqrt(static_cast<float>(c.D));

    std::vector<float> hQ(c.N * c.D), hK(c.N * c.D), hV(c.N * c.D);
    std::vector<float> hO(c.N * c.D), hRef(c.N * c.D);

    std::mt19937 rng(0xBEEF);
    // Moderate range; attention post-scale stays in sensible numeric area.
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (auto& v : hQ) v = dist(rng);
    for (auto& v : hK) v = dist(rng);
    for (auto& v : hV) v = dist(rng);

    float *dQ, *dK, *dV, *dO, *dWs;
    OPLITE_CUDA_CHECK(cudaMalloc(&dQ,  qkv_bytes));
    OPLITE_CUDA_CHECK(cudaMalloc(&dK,  qkv_bytes));
    OPLITE_CUDA_CHECK(cudaMalloc(&dV,  qkv_bytes));
    OPLITE_CUDA_CHECK(cudaMalloc(&dO,  qkv_bytes));
    OPLITE_CUDA_CHECK(cudaMalloc(&dWs, ws_bytes));
    OPLITE_CUDA_CHECK(cudaMemcpy(dQ, hQ.data(), qkv_bytes, cudaMemcpyHostToDevice));
    OPLITE_CUDA_CHECK(cudaMemcpy(dK, hK.data(), qkv_bytes, cudaMemcpyHostToDevice));
    OPLITE_CUDA_CHECK(cudaMemcpy(dV, hV.data(), qkv_bytes, cudaMemcpyHostToDevice));

    unfused_attention_forward(dQ, dK, dV, dO, dWs, c.N, c.D, scale);
    OPLITE_CUDA_KERNEL_CHECK();
    OPLITE_CUDA_CHECK(cudaDeviceSynchronize());

    OPLITE_CUDA_CHECK(cudaMemcpy(hO.data(), dO, qkv_bytes, cudaMemcpyDeviceToHost));

    // CPU reference.
    cpu_attention_forward(hQ.data(), hK.data(), hV.data(), hRef.data(),
                          c.N, c.D, scale);

    // Loose tolerance: attention accumulates a length-N dot product
    // after softmax; for N=1024 the FP32 error budget is ~1e-3.
    bool ok = allclose(hO.data(), hRef.data(), c.N * c.D,
                       /*rtol=*/1e-2f, /*atol=*/1e-3f);
    std::printf("  %-22s  N=%-5d D=%-4d  %s\n",
                c.name, c.N, c.D, ok ? "PASS" : "FAIL");

    cudaFree(dQ);
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(dO);
    cudaFree(dWs);
    return ok;
}

} // namespace

int main() {
    // N must be multiple of 128 (GEMM tile BM/BN).
    // D must be multiple of 128 for the P @ V step (same reason).
    //
    // Start small: CPU reference is O(N^2 * D) ~ N^2 * 128 ops, so
    // N=1024 is ~128M ops and finishes in <1s.
    const Case cases[] = {
        {"tiny",    128, 128},
        {"small",   256, 128},
        {"medium",  512, 128},
        {"large",  1024, 128},
        // {"big",   2048, 128},   // CPU ref takes ~4s; enable on demand
    };

    int failed = 0;
    std::printf("[unfused_attention_forward]\n");
    for (const auto& c : cases) failed += !run_one(c);

    std::printf("\n%s (%d failures)\n",
                failed == 0 ? "ALL PASSED" : "FAILED", failed);
    return failed == 0 ? 0 : 1;
}
