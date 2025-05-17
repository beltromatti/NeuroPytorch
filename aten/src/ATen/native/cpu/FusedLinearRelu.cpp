#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <c10/util/irange.h>
#include <cmath>

namespace at {
namespace native {

Tensor fused_linear_relu_cpu(const Tensor& input, const Tensor& weight, const Tensor& bias) {
  TORCH_CHECK(input.device().is_cpu(), "input must be on CPU");
  TORCH_CHECK(weight.device().is_cpu(), "weight must be on CPU");
  TORCH_CHECK(bias.device().is_cpu(), "bias must be on CPU");

  auto M = input.size(0);
  auto K = input.size(1);
  auto N = weight.size(0);

  auto output = at::empty({M, N}, input.options());

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_linear_relu_cpu", [&] {
    auto input_ptr = input.data_ptr<scalar_t>();
    auto weight_ptr = weight.data_ptr<scalar_t>();
    auto bias_ptr = bias.data_ptr<scalar_t>();
    auto output_ptr = output.data_ptr<scalar_t>();

    at::parallel_for(0, M, 0, [&](int64_t start, int64_t end) {
      for (int64_t i = start; i < end; ++i) {
        for (int64_t j = 0; j < N; ++j) {
          scalar_t acc = 0;
          for (int64_t k = 0; k < K; ++k) {
            acc += input_ptr[i * K + k] * weight_ptr[j * K + k];
          }
          acc += bias_ptr[j];
          output_ptr[i * N + j] = acc > 0 ? acc : 0; // ReLU
        }
      }
    });
  });

  return output;
}

}} // namespace at::native
