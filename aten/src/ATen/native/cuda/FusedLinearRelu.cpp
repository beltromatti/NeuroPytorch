#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

namespace at {
namespace native {

Tensor fused_linear_relu_cuda(const Tensor& input, const Tensor& weight, const Tensor& bias) {
  const int M = input.size(0);
  const int K = input.size(1);
  const int N = weight.size(0);

  auto output = at::empty({M, N}, input.options());

  dim3 threads(16, 16);
  dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_linear_relu_cuda", [&] {
    fused_linear_relu_kernel<scalar_t><<<blocks, threads>>>(
      input.data_ptr<scalar_t>(),
      weight.data_ptr<scalar_t>(),
      bias.data_ptr<scalar_t>(),
      output.data_ptr<scalar_t>(),
      M, N, K);
  });

  return output;
}

}} // namespace at::native