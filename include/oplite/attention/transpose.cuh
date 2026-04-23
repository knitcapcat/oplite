#pragma once
//
// 2D transpose:  in[rows, cols]  ->  out[cols, rows]  (both row-major).
//
// Used in unfused attention to produce K^T before calling a standard
// A @ B GEMM for the Q @ K^T step.
//
// Algorithm: classic tiled transpose with shared memory.
//   1. Each block reads a BS x BS tile from GMEM, coalesced.
//   2. Writes it to shared memory.
//   3. __syncthreads.
//   4. Reads the tile in transposed order from SMEM.
//   5. Writes to GMEM, still coalesced because we swap block coords
//      so the write stride is along the contiguous dim.
//
// Why the [BS][BS+1] padding:
//   The write phase does  tile[threadIdx.x][threadIdx.y]  --- that's a
//   column access. Without padding, a warp of 32 threads reading the
//   same column would all hit the same bank (bank = col % 32), giving
//   32-way conflict. Padding the inner dim to BS+1 makes the stride
//   coprime with 32, spreading the access over all banks.
//
//   This is the "inter-row" bank-conflict fix you learned earlier.
//
#include <cuda_runtime.h>
#include "oplite/common.cuh"

namespace oplite {

template <int BS>
__global__ void transpose_2d_kernel(
    const float* __restrict__ input,   // [rows, cols] row-major
    float* __restrict__ output,        // [cols, rows] row-major
    int rows, int cols)
{
    __shared__ float tile[BS][BS + 1];   // +1 breaks bank conflicts

    // Read phase: block (blockIdx.x, blockIdx.y) reads a BS x BS
    // tile starting at (blockIdx.y * BS, blockIdx.x * BS) in input.
    int in_row = blockIdx.y * BS + threadIdx.y;
    int in_col = blockIdx.x * BS + threadIdx.x;
    if (in_row < rows && in_col < cols) {
        tile[threadIdx.y][threadIdx.x] = input[in_row * cols + in_col];
    }
    __syncthreads();

    // Write phase: block coords are swapped, so the destination tile
    // is at (blockIdx.x * BS, blockIdx.y * BS) in output, which is
    // exactly where the transposed data should live.
    int out_row = blockIdx.x * BS + threadIdx.y;
    int out_col = blockIdx.y * BS + threadIdx.x;
    if (out_row < cols && out_col < rows) {
        // Read tile in transposed order: tile[x][y] instead of [y][x].
        output[out_row * rows + out_col] = tile[threadIdx.x][threadIdx.y];
    }
}

template <int BS = 32>
inline void launch_transpose_2d(
    const float* input, float* output,
    int rows, int cols,
    cudaStream_t stream = nullptr)
{
    dim3 block(BS, BS);
    dim3 grid(ceil_div(cols, BS), ceil_div(rows, BS));
    transpose_2d_kernel<BS><<<grid, block, 0, stream>>>(
        input, output, rows, cols);
}

} // namespace oplite
