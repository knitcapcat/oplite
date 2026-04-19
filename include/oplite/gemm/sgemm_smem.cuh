#pragma once
//
// 4.2 Block-tiled SGEMM with shared memory.
//
// Each block produces a BS x BS output tile of C. Inside the K loop the
// block cooperatively loads a BS x BS tile of A and a BS x BS tile of B
// into shared memory, then every thread accumulates one output element
// from those tiles.
//
// Why this beats naive:
//   - Each loaded A element is reused BS times (across columns)
//   - Each loaded B element is reused BS times (across rows)
//   - DRAM traffic drops by ~BS compared to naive (cache-independent)
//
// Compared to register-tiled GEMM this is still SMEM-bound (every FMA
// reads two SMEM operands). It exists in this library mainly as a
// pedagogical step between naive and register-tiled.
//
#include <cuda_runtime.h>
#include "oplite/common.cuh"

namespace oplite {

template <int BS = 32>
__global__ void sgemm_smem_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    __shared__ float As[BS * BS];
    __shared__ float Bs[BS * BS];

    const int tx = threadIdx.x;        // 0..BS-1, maps to column
    const int ty = threadIdx.y;        // 0..BS-1, maps to row
    const int row = blockIdx.y * BS + ty;
    const int col = blockIdx.x * BS + tx;

    float acc = 0.0f;

    for (int k_tile = 0; k_tile < K; k_tile += BS) {
        // Cooperative load. Out-of-bounds positions are zeroed so they
        // do not contribute to the dot product.
        int a_col = k_tile + tx;
        int b_row = k_tile + ty;
        As[ty * BS + tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[ty * BS + tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BS; ++kk) {
            acc += As[ty * BS + kk] * Bs[kk * BS + tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

template <int BS = 32>
inline void launch_sgemm_smem(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    cudaStream_t stream = nullptr)
{
    dim3 block(BS, BS);
    dim3 grid(ceil_div(N, BS), ceil_div(M, BS));
    sgemm_smem_kernel<BS><<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

} // namespace oplite
