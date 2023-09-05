// modify  common_cuda_helper.hpp in mmcv.ops
#ifndef GRID_SAMPLE_HELPER
#define GRID_SAMPLE_HELPER
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include <cmath>
#include <limits>


namespace grid_sample {
  enum class GridSamplerInterpolation {Nearest, Bilinear}; // cublic not implemented
  enum class GridSamplerPadding {Zeros, Border}; // border and reflect is not implemented

  const int MAXTENSORDIMS = 10;
  struct TensorDesc {
  int shape[MAXTENSORDIMS];
  int stride[MAXTENSORDIMS];
  int dim;
  };
}


// same with 
// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/cuda/detail/KernelUtils.h#L14
#define CUDA_1D_KERNEL_LOOP(i, n)                              \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

#define CUDA_2D_KERNEL_LOOP(i, n, j, m)                             \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);   \
       i += blockDim.x * gridDim.x)                                 \
    for (size_t j = blockIdx.y * blockDim.y + threadIdx.y; j < (m); \
         j += blockDim.y * gridDim.y)

#define CUDA_2D_KERNEL_BLOCK_LOOP(i, n, j, m)          \
  for (size_t i = blockIdx.x; i < (n); i += gridDim.x) \
    for (size_t j = blockIdx.y; j < (m); j += gridDim.y)

#define THREADS_PER_BLOCK 512
#define THREADS_PER_BLOCK_SCALE 32
#define THREADS_PER_BLOCK_SHIFT 32

#define CUDA_CHECK(condition) {GpuAssert((condition), __FILE__, __LINE__);}
inline void GpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) { exit(code); }
  }
}

inline int GET_BLOCKS(const int N, const int num_threads = THREADS_PER_BLOCK) {
  int optimal_block_num = (N - 1) / num_threads + 1;
  int max_block_num = 4096;
  return min(optimal_block_num, max_block_num);
}

// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/cuda/detail/KernelUtils.h
// inline int GET_BLOCKS(const int N, const int num_threads = THREADS_PER_BLOCK) {
//   auto block_num = (N-1) / num_threads + 1;
//   int max_block_num = std::numeric_limits<int>::max();
//   return min(block_num, max_block_num);
// }

#endif  // GRID_SAMPLE_HELPER
