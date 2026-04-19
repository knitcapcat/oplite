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
