# oplite

A from-scratch CUDA kernel library for learning. The end goal is a
working Paged Attention kernel, with stops at GEMM, online softmax,
Flash Attention 2, and CuTe along the way.

Kernels are header-only C++ templates so tile sizes and dtypes can be
specialized at compile time, mirroring the style of CUTLASS.

## Layout

```
oplite/
├── include/oplite/
│   ├── common.cuh                      # CUDA_CHECK, ceil_div, allclose, ...
│   └── gemm/
│       ├── sgemm_naive.cuh             # 4.1 baseline (no shared memory)
│       ├── sgemm_smem.cuh              # 4.2 block-tiled (shared memory)
│       └── sgemm_reg_tiled.cuh         # 4.3 register-tiled + float4 load
├── tests/test_sgemm.cu                 # cuBLAS as numerical oracle
└── bench/bench_sgemm.cu                # GFLOPS table vs cuBLAS
```

## Build

Defaults to `sm_80` (A100). Override with `-DCMAKE_CUDA_ARCHITECTURES=<arch>`.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Correctness
./build/tests/test_sgemm

# Performance
./build/bench/bench_sgemm
```

## Adding a new kernel

1. Drop a templated header under `include/oplite/<op>/<kernel_name>.cuh`.
   Provide both `<kernel_name>_kernel` (the `__global__`) and
   `launch_<kernel_name>` (a host wrapper that picks the launch config).
2. Include it in `tests/test_<op>.cu` and add an entry to the test list.
3. Include it in `bench/bench_<op>.cu` and add an entry to the bench table.

## Profiling

Use `ncu` to compare kernels:

```bash
ncu --set full --kernel-name regex:sgemm_.* \
    --print-summary per-kernel ./build/bench/bench_sgemm
```

For a specific kernel, replace the regex with the mangled name (use
`cuobjdump --dump-elf-symbols` if you need it).
