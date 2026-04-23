#pragma once
//
// Unfused attention forward (single head, FP32).
//
//   S = scale * Q @ K^T          shape [N, N]
//   P = row_softmax(S)           shape [N, N]
//   O = P @ V                    shape [N, D]
//
// Call sequence:
//   1. transpose_2d      K[N,D] -> K^T[D,N]   (workspace)
//   2. sgemm_cp_async    Q[N,D] @ K^T[D,N] = S[N,N]   (workspace)
//   3. softmax_online    S -> P   (in-place, with fused scale)
//   4. sgemm_cp_async    P[N,N] @ V[N,D] = O[N,D]
//
// Intermediate tensors (K^T and S/P) live in a caller-provided
// workspace in GMEM. Not using cudaMalloc/Free inside the op means:
//   - no fragmentation from repeated calls
//   - the memory footprint is explicit and auditable in benchmarks
//   - the three intermediate GMEM matrices make the "why FA fuses"
//     story concrete: we'll see the cost in ncu
//
// Shape constraints (driven by the underlying GEMM's tile defaults
// BM = BN = 128, BK = 8):
//   - N must be a multiple of 128
//   - D must be a multiple of 128 (usually 128 exactly for modern LLMs)
//   - If you need D = 64, a different GEMM template instantiation is
//     required for the P @ V step; not supported in this first version.
//
#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
#include "oplite/common.cuh"
#include "oplite/gemm/sgemm_cp_async.cuh"
#include "oplite/softmax/softmax_online.cuh"
#include "oplite/attention/transpose.cuh"

namespace oplite {

// Workspace is laid out as:
//   [0            .. D*N)             K^T  [D, N]
//   [D*N          .. D*N + N*N)       S/P  [N, N]  (softmax is in-place)
//
// Total size in floats:
inline size_t unfused_attention_forward_workspace_floats(int N, int D) {
    return static_cast<size_t>(D) * N
         + static_cast<size_t>(N) * N;
}

inline size_t unfused_attention_forward_workspace_bytes(int N, int D) {
    return unfused_attention_forward_workspace_floats(N, D) * sizeof(float);
}

inline void unfused_attention_forward(
    const float* Q,        // [N, D] row-major
    const float* K,        // [N, D] row-major
    const float* V,        // [N, D] row-major
    float* O,              // [N, D] row-major (output)
    float* workspace,      // size = unfused_attention_forward_workspace_bytes(N, D)
    int N, int D,
    float scale = 0.0f,    // pass 0 to default to 1/sqrt(D)
    cudaStream_t stream = nullptr)
{
    if (scale == 0.0f) {
        scale = 1.0f / std::sqrt(static_cast<float>(D));
    }

    // Slice the workspace.
    float* K_T = workspace;                                        // [D, N]
    float* S   = workspace + static_cast<size_t>(D) * N;           // [N, N]

    // 1. K_T = transpose(K)
    launch_transpose_2d<>(K, K_T, /*rows=*/N, /*cols=*/D, stream);

    // 2. S = Q @ K_T      (M=N, N=N, K=D)
    launch_sgemm_cp_async<>(Q, K_T, S, /*M=*/N, /*N=*/N, /*K=*/D, stream);

    // 3. P = softmax(scale * S)   in-place (S and P share storage)
    launch_softmax_online<>(S, S, /*M=*/N, /*N=*/N, scale, stream);

    // 4. O = P @ V       (M=N, N=D, K=N)
    launch_sgemm_cp_async<>(S, V, O, /*M=*/N, /*N=*/D, /*K=*/N, stream);
}

} // namespace oplite
