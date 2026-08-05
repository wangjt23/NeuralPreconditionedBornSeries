#include <torch/extension.h>

torch::Tensor circular_conv2d_cuda(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    int64_t stride);

torch::Tensor fused_scale_gate_cuda(
    const torch::Tensor& x,
    const torch::Tensor& coeff,
    const torch::Tensor& local_weight,
    const torch::Tensor& local_bias,
    const torch::Tensor& global_bias,
    const torch::Tensor& matrix,
    double bounded_gain,
    double mix_gain);

torch::Tensor circular_conv2d(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    int64_t stride) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
  TORCH_CHECK(!bias.defined() || bias.is_cuda(), "bias must be a CUDA tensor");
  return circular_conv2d_cuda(input, weight, bias, stride);
}

torch::Tensor fused_scale_gate(
    const torch::Tensor& x,
    const torch::Tensor& coeff,
    const torch::Tensor& local_weight,
    const torch::Tensor& local_bias,
    const torch::Tensor& global_bias,
    const torch::Tensor& matrix,
    double bounded_gain,
    double mix_gain) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(coeff.is_cuda(), "coeff must be a CUDA tensor");
  TORCH_CHECK(local_weight.is_cuda(), "local_weight must be a CUDA tensor");
  TORCH_CHECK(!local_bias.defined() || local_bias.is_cuda(), "local_bias must be a CUDA tensor");
  TORCH_CHECK(global_bias.is_cuda(), "global_bias must be a CUDA tensor");
  TORCH_CHECK(!matrix.defined() || matrix.is_cuda(), "matrix must be a CUDA tensor");
  return fused_scale_gate_cuda(
      x, coeff, local_weight, local_bias, global_bias, matrix, bounded_gain, mix_gain);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("circular_conv2d", &circular_conv2d, "Periodic circular convolution (CUDA)");
  m.def("fused_scale_gate", &fused_scale_gate, "Scale-aware gate with optional channel mix (CUDA)");
}
