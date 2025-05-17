#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void fused_linear_relu_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int M, int N, int K) {

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    scalar_t acc = 0;
    for (int i = 0; i < K; ++i) {
      acc += input[row * K + i] * weight[col * K + i];
    }
    acc += bias[col];
    output[row * N + col] = acc > 0 ? acc : 0;  // ReLU
  }
}