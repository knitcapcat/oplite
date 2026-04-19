#pragma once

#include <cuda_runtime.h>
#include "oplite/common.cuh"

namespace oplite {

template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_vec_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    constexpr int kVec             = 4;
    constexpr int kThreadsPerBlock = (BM / TM) * (BN / TN);
    constexpr int kThreadColCount  = BN / TN;
    constexpr int kAVecPerRow      = BK / kVec;   // float4s per A-tile row
    constexpr int kBVecPerRow      = BN / kVec;   // float4s per B-tile row

    static_assert(BM % TM == 0 && BN % TN == 0,
                  "Block tile must be divisible by thread tile.");
    static_assert(BK % kVec == 0 && BN % kVec == 0,
                  "BK and BN must be multiples of 4 for float4 GMEM->SMEM loads.");
    static_assert(TN % kVec == 0,
                  "TN must be a multiple of 4 for float4 register->GMEM stores.");
    static_assert(TM % kVec == 0,
                  "TM must be a multiple of 4 for float4 SMEM->register A loads "
                  "(requires transposed As layout).");
    static_assert(kThreadsPerBlock == BM * kAVecPerRow,
                  "Thread count must match A-tile float4 count "
                  "(one float4 per thread per K-step).");
    static_assert(kThreadsPerBlock == BK * kBVecPerRow,
                  "Thread count must match B-tile float4 count "
                  "(one float4 per thread per K-step).");

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // Where this block's output tile starts in C.
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    // Where this thread's output tile starts inside the block tile.
    const int thread_row = threadIdx.x / kThreadColCount;
    const int thread_col = threadIdx.x % kThreadColCount;

    // GMEM-load coordinates. Each thread owns exactly one float4 of A
    // and one float4 of B per K-step.
    const int load_row_a = threadIdx.x / kAVecPerRow;
    const int load_col_a = (threadIdx.x % kAVecPerRow) * kVec;
    const int load_row_b = threadIdx.x / kBVecPerRow;
    const int load_col_b = (threadIdx.x % kBVecPerRow) * kVec;

    float reg_a[TM];
    float reg_b[TN];
    float acc[TM * TN] = {0.0f};

    // Move base pointers to the top-left of this block's input slabs.
    // We then advance them by BK / BK*N each iteration (see end of loop).
    A += block_row * K;
    B += block_col;

    for (int k_tile = 0; k_tile < K; k_tile += BK) {
        // ----- GMEM -> SMEM (one float4 per thread for each tile) -----
        const float4 a4 = *reinterpret_cast<const float4*>(
            &A[load_row_a * K + load_col_a]);
        As[(load_col_a + 0) * BM + load_row_a] = a4.x;
        As[(load_col_a + 1) * BM + load_row_a] = a4.y;
        As[(load_col_a + 2) * BM + load_row_a] = a4.z;
        As[(load_col_a + 3) * BM + load_row_a] = a4.w;

        const float4 b4 = *reinterpret_cast<const float4*>(
            &B[load_row_b * N + load_col_b]);
        *reinterpret_cast<float4*>(
            &Bs[load_row_b * BN + load_col_b]) = b4;

        __syncthreads();

        // Advance for the next iteration. (Doing this before compute is
        // safe because the loads above are already complete in SMEM.)
        A += BK;        // walk BK columns to the right inside this row strip
        B += BK * N;    // walk BK rows down inside this column strip

        // ----- SMEM -> registers -> FMA -----
        // As layout is transposed: As[BK][BM] (kk is the row, M is the
        // contiguous dimension). So reading TM consecutive elements of
        // A for fixed kk is now a contiguous SMEM access -> LDS.128.
        // Bs is unchanged: Bs[BK][BN], and we read TN consecutive
        // elements of B for fixed kk -> also LDS.128.
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            #pragma unroll
            for (int i = 0; i < TM; i += kVec) {
                *reinterpret_cast<float4*>(&reg_a[i]) =
                    *reinterpret_cast<const float4*>(
                        &As[kk * BM + thread_row * TM + i]);
            }
            #pragma unroll
            for (int j = 0; j < TN; j += kVec) {
                *reinterpret_cast<float4*>(&reg_b[j]) =
                    *reinterpret_cast<const float4*>(
                        &Bs[kk * BN + thread_col * TN + j]);
            }
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i * TN + j] += reg_a[i] * reg_b[j];
                }
            }
        }

        __syncthreads();
    }

    // ----- Write back. No bounds check yet; the launcher requires
    // M and N to be multiples of BM and BN. Padding support belongs
    // in a separate variant. -----
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int row = block_row + thread_row * TM + i;
        int col_base = block_col + thread_col * TN;
        #pragma unroll
        for (int j = 0; j < TN; j += 4) {
            float4 v;
            v.x = acc[i * TN + j + 0];
            v.y = acc[i * TN + j + 1];
            v.z = acc[i * TN + j + 2];
            v.w = acc[i * TN + j + 3];
            *reinterpret_cast<float4*>(&C[row * N + col_base + j]) = v;
        }
    }
}

template <int BM = 128, int BN = 128, int BK = 8, int TM = 8, int TN = 8>
inline void launch_sgemm_vec(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    cudaStream_t stream = nullptr)
{
    constexpr int kThreadsPerBlock = (BM / TM) * (BN / TN);
    dim3 block(kThreadsPerBlock);
    dim3 grid(N / BN, M / BM);
    sgemm_vec_kernel<BM, BN, BK, TM, TN><<<grid, block, 0, stream>>>(
        A, B, C, M, N, K);
}

} // namespace oplite
