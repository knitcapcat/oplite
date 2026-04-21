#pragma once
//
// 4.7 SGEMM with cp.async software pipelining.
//
// Same math, same tile geometry, same transposed-A + vectorized-B
// SMEM layout as sgemm_vec.cuh. The only thing that changes is
// the GMEM -> SMEM path:
//
//   Old (sgemm_vec):  LDG -> register -> STS     (synchronous,
//                                                 blocks compute)
//
//   New (cp_async):   cp.async  GMEM -> SMEM     (asynchronous,
//                                                 compute runs
//                                                 while next tile
//                                                 loads in flight)
//
// Structure (2-stage ping-pong, single async group in flight):
//
//   Prologue:   issue load(0)   commit
//   Main loop k = 0 .. K/BK - 2:
//     issue load(k+1)   commit             <-- 2 groups now pending
//     wait_group<1>   __syncthreads()       <-- keep k+1 in flight,
//                                              only wait for k
//     compute using stage k & 1
//   Epilogue:  wait_group<0>   __syncthreads()
//              compute last tile
//
// This overlaps one tile's compute with the next tile's GMEM load.
// A 3-stage pipeline (depth-2 in flight) is a straightforward
// extension once this works.
//
#include <cuda_runtime.h>
#include "oplite/common.cuh"

namespace oplite {

template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_cp_async_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    constexpr int kVec             = 4;
    constexpr int kThreadsPerBlock = (BM / TM) * (BN / TN);
    constexpr int kThreadColCount  = BN / TN;
    constexpr int kAVecPerRow      = BK / kVec;
    constexpr int kBVecPerRow      = BN / kVec;

    static_assert(BM % TM == 0 && BN % TN == 0,
                  "Block tile must be divisible by thread tile.");
    static_assert(BK % kVec == 0 && BN % kVec == 0,
                  "BK and BN must be multiples of 4 for float4 layouts.");
    static_assert(TN % kVec == 0, "TN must be a multiple of 4.");
    static_assert(TM % kVec == 0, "TM must be a multiple of 4.");
    static_assert(kThreadsPerBlock == BM * kAVecPerRow,
                  "Thread count must match A-tile float4 count.");
    static_assert(kThreadsPerBlock == BK * kBVecPerRow,
                  "Thread count must match B-tile float4 count.");

    // Two-stage ping-pong SMEM. Each stage holds one K-tile's data.
    __shared__ float As[2][BK * BM];   // transposed: [BK][BM]
    __shared__ float Bs[2][BK * BN];   // straight:   [BK][BN]

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    const int thread_row = threadIdx.x / kThreadColCount;
    const int thread_col = threadIdx.x % kThreadColCount;

    const int load_row_a = threadIdx.x / kAVecPerRow;
    const int load_col_a = (threadIdx.x % kAVecPerRow) * kVec;
    const int load_row_b = threadIdx.x / kBVecPerRow;
    const int load_col_b = (threadIdx.x % kBVecPerRow) * kVec;

    float reg_a[TM];
    float reg_b[TN];
    float acc[TM * TN] = {0.0f};

    // Position GMEM base pointers at this block's row/col strip.
    A += block_row * K;
    B += block_col;

    // ------------------------------------------------------------
    // Helper: issue async loads for the K-tile starting at k_base
    // into the given SMEM stage. Returns immediately; data becomes
    // valid only after cp_async_wait_group<N>() + __syncthreads().
    // ------------------------------------------------------------
    auto issue_load_tile = [&](int k_base, int stage) {
        // --- A: 4 scalar cp.async to preserve the transpose ---
        // Source: A[load_row_a][k_base + load_col_a..+3]  (contiguous)
        // Dest:   As[stage][(load_col_a + d) * BM + load_row_a] for d=0..3
        const float* a_src = &A[load_row_a * K + k_base + load_col_a];
        cp_async_ca_4(&As[stage][(load_col_a + 0) * BM + load_row_a], a_src + 0);
        cp_async_ca_4(&As[stage][(load_col_a + 1) * BM + load_row_a], a_src + 1);
        cp_async_ca_4(&As[stage][(load_col_a + 2) * BM + load_row_a], a_src + 2);
        cp_async_ca_4(&As[stage][(load_col_a + 3) * BM + load_row_a], a_src + 3);

        // --- B: 1 vectorized cp.async.16 (no transpose) ---
        const float* b_src = &B[(k_base + load_row_b) * N + load_col_b];
        cp_async_cg_16(&Bs[stage][load_row_b * BN + load_col_b], b_src);
    };

    // ------------------------------------------------------------
    // Helper: the inner compute loop, unchanged from sgemm_vec,
    // parameterized by which SMEM stage to read from.
    // ------------------------------------------------------------
    auto compute_tile = [&](int stage) {
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            #pragma unroll
            for (int i = 0; i < TM; i += kVec) {
                *reinterpret_cast<float4*>(&reg_a[i]) =
                    *reinterpret_cast<const float4*>(
                        &As[stage][kk * BM + thread_row * TM + i]);
            }
            #pragma unroll
            for (int j = 0; j < TN; j += kVec) {
                *reinterpret_cast<float4*>(&reg_b[j]) =
                    *reinterpret_cast<const float4*>(
                        &Bs[stage][kk * BN + thread_col * TN + j]);
            }
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i * TN + j] += reg_a[i] * reg_b[j];
                }
            }
        }
    };

    // ============================================================
    // Prologue: kick off the load for k_tile = 0
    // ============================================================
    int stage = 0;
    issue_load_tile(0, stage);
    cp_async_commit_group();

    // ============================================================
    // Main loop: overlap compute(k) with load(k+1)
    // ============================================================
    for (int k_tile = 0; k_tile + BK < K; k_tile += BK) {
        const int next_stage = 1 - stage;

        // Launch next tile's load BEFORE computing current.
        // Now 2 commit groups are pending in flight.
        issue_load_tile(k_tile + BK, next_stage);
        cp_async_commit_group();

        // Wait for the CURRENT tile (leave the newer one pending).
        // wait_group<1> = "at most 1 group still pending" = current is done.
        cp_async_wait_group<1>();
        __syncthreads();

        compute_tile(stage);

        stage = next_stage;
    }

    // ============================================================
    // Epilogue: last tile. Its load was issued either in the
    // prologue (if K == BK) or in the last loop iteration.
    // ============================================================
    cp_async_wait_group<0>();
    __syncthreads();
    compute_tile(stage);

    // ============================================================
    // Store back: identical to sgemm_vec (float4 vectorized).
    // ============================================================
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int row = block_row + thread_row * TM + i;
        int col_base = block_col + thread_col * TN;
        #pragma unroll
        for (int j = 0; j < TN; j += kVec) {
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
inline void launch_sgemm_cp_async(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    cudaStream_t stream = nullptr)
{
    constexpr int kThreadsPerBlock = (BM / TM) * (BN / TN);
    dim3 block(kThreadsPerBlock);
    dim3 grid(N / BN, M / BM);
    sgemm_cp_async_kernel<BM, BN, BK, TM, TN><<<grid, block, 0, stream>>>(
        A, B, C, M, N, K);
}

} // namespace oplite
