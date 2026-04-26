// Benchmarks unfused attention forward.
// Reports wall time, TFLOPS, and HBM traffic.
//
// The printed "HBM overhead ratio" is the key metric for this stage:
// it shows how much more memory the unfused approach reads/writes
// compared to what Flash Attention must touch at minimum.
//
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>

#include "oplite/common.cuh"
#include "oplite/attention/unfused_forward.cuh"

using namespace oplite;

namespace {

constexpr int kWarmup = 3;
constexpr int kIter   = 20;

void run_size(int N, int D) {
    const size_t qkv_bytes = static_cast<size_t>(N) * D * sizeof(float);
    const size_t ws_bytes  = unfused_attention_forward_workspace_bytes(N, D);

    std::vector<float> hQ(N * D), hK(N * D), hV(N * D);
    std::mt19937 rng(0xBEEF);
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

    // Warmup
    for (int i = 0; i < kWarmup; ++i) {
        unfused_attention_forward(dQ, dK, dV, dO, dWs, N, D);
    }
    OPLITE_CUDA_CHECK(cudaDeviceSynchronize());

    // Time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < kIter; ++i) {
        unfused_attention_forward(dQ, dK, dV, dO, dWs, N, D);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_total = 0.0f;
    cudaEventElapsedTime(&ms_total, start, stop);
    const float ms = ms_total / kIter;

    // ---------- Math: FLOPs for one forward pass ----------
    // Q @ K^T: 2 * N * N * D
    // softmax: ~3 * N * N (not counted; negligible)
    // P @ V:   2 * N * N * D
    // Total:  ~4 * N * N * D
    const double flops = 4.0 * N * N * D;
    const double tflops = flops / (ms * 1e9);

    // ---------- Memory traffic ledger (all in floats) ----------
    // Step 1 transpose:      read K (ND) + write K^T (ND)         = 2 N D
    // Step 2 Q@K^T:          read Q (ND) + read K^T (ND) + write S (NN) = 2 N D + N*N
    // Step 3 softmax inplace:read S (NN) + write P (NN)           = 2 N*N
    // Step 4 P@V:            read P (NN) + read V (ND) + write O (ND)  = 2 N D + N*N
    // -----------------------------------------------------------
    //                                              total          = 6 N D + 4 N*N
    const double unfused_floats =
        6.0 * static_cast<double>(N) * D + 4.0 * static_cast<double>(N) * N;
    const double unfused_bytes = unfused_floats * 4.0;

    // Flash Attention's theoretical minimum HBM traffic:
    //   read Q, read K, read V, write O  = 4 N D floats
    const double fa_min_floats = 4.0 * static_cast<double>(N) * D;
    const double fa_min_bytes  = fa_min_floats * 4.0;

    const double overhead_ratio = unfused_bytes / fa_min_bytes;
    const double achieved_bw_gbs = unfused_bytes / (ms * 1e6);

    std::printf(
        "N=%-5d D=%-3d  "
        "%.3f ms  %.1f TF  "
        "HBM=%6.1f MB  FA_min=%5.1f MB  "
        "overhead=%.1fx  "
        "achieved=%.0f GB/s\n",
        N, D, ms, tflops,
        unfused_bytes / (1024.0 * 1024.0),
        fa_min_bytes  / (1024.0 * 1024.0),
        overhead_ratio,
        achieved_bw_gbs);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dQ);
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(dO);
    cudaFree(dWs);
}

} // namespace

int main() {
    std::printf("=== Unfused attention forward (FP32, D=128) ===\n");
    // N must be multiple of 128 for our GEMM tile constraints.
    const int Ns[] = {1024, 2048, 4096, 8192};
    for (int N : Ns) run_size(N, 128);
    return 0;
}
