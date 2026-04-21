#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

namespace oplite {

#define OPLITE_CUDA_CHECK(expr)                                              \
    do {                                                                     \
        cudaError_t _err = (expr);                                           \
        if (_err != cudaSuccess) {                                           \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",             \
                         cudaGetErrorName(_err), __FILE__, __LINE__,         \
                         cudaGetErrorString(_err));                          \
            std::abort();                                                    \
        }                                                                    \
    } while (0)

// Check the last async error from a kernel launch. Call right after a launch.
#define OPLITE_CUDA_KERNEL_CHECK()                                           \
    do {                                                                     \
        OPLITE_CUDA_CHECK(cudaGetLastError());                               \
    } while (0)

__host__ __device__ constexpr int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

// ============================================================
// cp.async helpers (Ampere+, i.e. sm_80 and later).
//
// cp.async initiates an asynchronous GMEM -> SMEM copy that does
// NOT go through registers. The copied data is only guaranteed to
// be visible after a cp.async.wait_group (plus a __syncthreads
// since other threads in the block may need the data too).
//
// Size must be 4, 8, or 16 bytes:
//   - 16B: both .ca (cache all, through L1) and .cg (cache global, L2 only) supported
//   - 4B / 8B: only .ca
//
// Prefer .cg for large streaming transfers (B tiles in GEMM) to
// save L1 pressure; .ca is required for small transfers (A tile
// scatter writes when transposing).
// ============================================================

// 16-byte async copy, L2-only caching.
__device__ __forceinline__
void cp_async_cg_16(void* smem_ptr, const void* gmem_ptr) {
    unsigned smem_int = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_int), "l"(gmem_ptr)
    );
}

// 4-byte async copy, L1+L2 caching (required for <16B).
__device__ __forceinline__
void cp_async_ca_4(void* smem_ptr, const void* gmem_ptr) {
    unsigned smem_int = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 4;\n"
        :: "r"(smem_int), "l"(gmem_ptr)
    );
}

// Mark all preceding cp.async operations as belonging to a new
// commit group. Groups serve as tracking units for wait_group.
__device__ __forceinline__
void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;\n" ::);
}

// Wait until at most N commit groups remain in flight.
// wait_group<0> means wait for ALL pending cp.async to finish.
template <int N>
__device__ __forceinline__
void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

// Numerical comparison helper for tests. Mimics torch.allclose semantics:
//   |a - b| <= atol + rtol * |b|
// Returns true if all elements pass; otherwise prints up to `max_report`
// mismatches and returns false.
inline bool allclose(const float* a, const float* b, int n,
                     float rtol = 1e-3f, float atol = 1e-3f,
                     int max_report = 5) {
    int bad = 0;
    for (int i = 0; i < n; ++i) {
        float diff = std::fabs(a[i] - b[i]);
        float thresh = atol + rtol * std::fabs(b[i]);
        if (!(diff <= thresh)) {
            if (bad < max_report) {
                std::fprintf(stderr,
                             "  mismatch at %d: got %g, ref %g (|diff|=%g, thresh=%g)\n",
                             i, a[i], b[i], diff, thresh);
            }
            ++bad;
        }
    }
    if (bad > 0) {
        std::fprintf(stderr, "  total mismatches: %d / %d\n", bad, n);
        return false;
    }
    return true;
}

} // namespace oplite
