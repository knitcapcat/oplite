#pragma once
//
// Row-wise online softmax (2-pass).
//
// Input:   x [M, N]  row-major
// Output:  y [M, N]  where y[m, n] = exp(x[m, n] - max_m) / sum_m
//
// Layout: one thread block per row. Each block:
//   Pass 1  online reduce of (m_i, s_i) pair using the coupled
//           rescaling rule:
//               s_new = s_old * exp(m_old - m_new) + exp(x - m_new)
//           This is the same update that lives inside Flash Attention's
//           inner loop; mastering it here means the FA2 softmax step
//           will already be familiar.
//   Pass 2  write y[m, n] = exp(x[m, n] - row_m) / row_s
//
// Block reduce: tree-reduce down to one warp, then warp-shuffle the
// final 32 values. Shared-memory usage is 2 * BLOCK_SIZE floats.
//
#include <cuda_runtime.h>
#include <cfloat>
#include "oplite/common.cuh"

namespace oplite {

template <int BLOCK_SIZE>
__global__ void softmax_online_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, float scale)
{
    static_assert(BLOCK_SIZE >= 32 && (BLOCK_SIZE & (BLOCK_SIZE - 1)) == 0,
                  "BLOCK_SIZE must be a power of two and at least 32.");

    const float* row_in  = input  + blockIdx.x * N;
    float*       row_out = output + blockIdx.x * N;

    // -------- Pass 1: online reduce to (row_m, row_s) --------
    // The `scale` factor is fused into the softmax input so attention
    // doesn't need a separate "scale" kernel (saves a GMEM round-trip).
    // Caller passes scale=1.0f when plain softmax is wanted.
    float local_m = -FLT_MAX;
    float local_s = 0.0f;
    for (int i = threadIdx.x; i < N; i += BLOCK_SIZE) {
        float x = row_in[i] * scale;
        float m_new = fmaxf(local_m, x);
        local_s = local_s * expf(local_m - m_new) + expf(x - m_new);
        local_m = m_new;
    }

    __shared__ float smem_m[BLOCK_SIZE];
    __shared__ float smem_s[BLOCK_SIZE];
    smem_m[threadIdx.x] = local_m;
    smem_s[threadIdx.x] = local_s;
    __syncthreads();

    // Tree reduce down to one warp (32 threads). Each step halves
    // the number of active threads, and combines (m, s) pairs with
    // the coupled rescaling rule.
    #pragma unroll
    for (int step = BLOCK_SIZE / 2; step >= 32; step >>= 1) {
        if (threadIdx.x < step) {
            float m1 = smem_m[threadIdx.x];
            float s1 = smem_s[threadIdx.x];
            float m2 = smem_m[threadIdx.x + step];
            float s2 = smem_s[threadIdx.x + step];
            float m_new = fmaxf(m1, m2);
            smem_s[threadIdx.x] = s1 * expf(m1 - m_new) + s2 * expf(m2 - m_new);
            smem_m[threadIdx.x] = m_new;
        }
        __syncthreads();
    }

    // Final 32-way reduce via warp shuffle (no __syncthreads needed
    // inside a warp).
    if (threadIdx.x < 32) {
        float m = smem_m[threadIdx.x];
        float s = smem_s[threadIdx.x];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float m2 = __shfl_down_sync(0xffffffff, m, offset);
            float s2 = __shfl_down_sync(0xffffffff, s, offset);
            float m_new = fmaxf(m, m2);
            s = s * expf(m - m_new) + s2 * expf(m2 - m_new);
            m = m_new;
        }
        if (threadIdx.x == 0) {
            smem_m[0] = m;
            smem_s[0] = s;
        }
    }
    __syncthreads();

    const float row_m = smem_m[0];
    const float row_s = smem_s[0];

    // -------- Pass 2: write y = exp(scale * x - row_m) / row_s --------
    // Note: input and output may alias (in-place softmax is supported,
    // since each thread reads and writes at independent indices).
    for (int i = threadIdx.x; i < N; i += BLOCK_SIZE) {
        row_out[i] = expf(row_in[i] * scale - row_m) / row_s;
    }
}

template <int BLOCK_SIZE = 256>
inline void launch_softmax_online(
    const float* input, float* output,
    int M, int N,
    float scale = 1.0f,
    cudaStream_t stream = nullptr)
{
    dim3 block(BLOCK_SIZE);
    dim3 grid(M);                    // one block per row
    softmax_online_kernel<BLOCK_SIZE><<<grid, block, 0, stream>>>(
        input, output, N, scale);
}

} // namespace oplite
