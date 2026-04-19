#pragma once
//
// 4.1 Naive SGEMM. One thread computes one C[i][j].
//
// All tensors are row-major:
//   A:  [M, K]
//   B:  [K, N]
//   C:  [M, N]
//
// Memory access pattern (with threadIdx.x mapped to column):
//   - B[k*N + col]: consecutive threads read consecutive cols  -> coalesced
//   - C[row*N + col]: same                                     -> coalesced
//   - A[row*K + k]:  threads with same row broadcast           -> L1-friendly
//
// This is intentionally simple. Use it as the perf baseline; expect
// SOL Memory near peak and SOL Compute very low.
//
#include <cuda_runtime.h>
#include "oplite/common.cuh"

namespace oplite {

template <int BLOCK_X = 16, int BLOCK_Y = 16>
__global__ void sgemm_naive_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    int row = blockIdx.y * BLOCK_Y + threadIdx.y;
    int col = blockIdx.x * BLOCK_X + threadIdx.x;

    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = acc;
}

template <int BLOCK_X = 16, int BLOCK_Y = 16>
inline void launch_sgemm_naive(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    cudaStream_t stream = nullptr)
{
    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid(ceil_div(N, BLOCK_X), ceil_div(M, BLOCK_Y));
    sgemm_naive_kernel<BLOCK_X, BLOCK_Y><<<grid, block, 0, stream>>>(
        A, B, C, M, N, K);
}

} // namespace oplite
