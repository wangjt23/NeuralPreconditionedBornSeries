#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace {

__global__ void circular_conv_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int64_t batch,
    int64_t in_channels,
    int64_t height,
    int64_t width,
    int64_t out_channels,
    int64_t kernel,
    int64_t stride,
    int64_t out_height,
    int64_t out_width,
    int has_bias) {
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = batch * out_channels * out_height * out_width;
  if (linear >= total) return;
  int64_t t = linear;
  int64_t ox = t % out_width;
  t /= out_width;
  int64_t oy = t % out_height;
  t /= out_height;
  int64_t oc = t % out_channels;
  int64_t b = t / out_channels;
  int64_t padding = kernel / 2;
  float sum = 0.0f;
  for (int64_t ic = 0; ic < in_channels; ++ic) {
    for (int64_t ky = 0; ky < kernel; ++ky) {
      int64_t iy = (oy * stride + ky - padding) % height;
      if (iy < 0) iy += height;
      for (int64_t kx = 0; kx < kernel; ++kx) {
        int64_t ix = (ox * stride + kx - padding) % width;
        if (ix < 0) ix += width;
        int64_t input_index = ((b * in_channels + ic) * height + iy) * width + ix;
        int64_t weight_index = ((oc * in_channels + ic) * kernel + ky) * kernel + kx;
        sum += input[input_index] * weight[weight_index];
      }
    }
  }
  if (has_bias) sum += bias[oc];
  output[linear] = sum;
}

__global__ void scale_gate_kernel(
    const float* x,
    const float* coeff,
    const float* local_weight,
    const float* local_bias,
    const float* global_bias,
    const float* matrix,
    float* output,
    int64_t batch,
    int64_t channels,
    int64_t coeff_channels,
    int64_t height,
    int64_t width,
    int has_local_bias,
    int has_matrix,
    float bounded_gain,
    float mix_gain) {
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = batch * channels * height * width;
  if (linear >= total) return;
  int64_t t = linear;
  int64_t px = t % width;
  t /= width;
  int64_t py = t % height;
  t /= height;
  int64_t channel = t % channels;
  int64_t b = t / channels;
  float local = 0.0f;
  for (int64_t k = 0; k < coeff_channels; ++k) {
    int64_t coeff_index = ((b * coeff_channels + k) * height + py) * width + px;
    local += coeff[coeff_index] * local_weight[channel * coeff_channels + k];
  }
  if (has_local_bias) local += local_bias[channel];
  local += global_bias[b * channels + channel];
  float result = x[linear] * (1.0f + bounded_gain * tanhf(local));
  if (has_matrix) {
    float mixed = 0.0f;
    for (int64_t other = 0; other < channels; ++other) {
      int64_t other_index = ((b * channels + other) * height + py) * width + px;
      int64_t matrix_index = (b * channels + channel) * channels + other;
      mixed += matrix[matrix_index] * x[other_index];
    }
    result += mix_gain * mixed / sqrtf(static_cast<float>(channels));
  }
  output[linear] = result;
}

}  // namespace

torch::Tensor circular_conv2d_cuda(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    int64_t stride) {
  TORCH_CHECK(input.scalar_type() == at::ScalarType::Float, "CUDA defect2 operator currently uses float32");
  TORCH_CHECK(weight.scalar_type() == at::ScalarType::Float, "CUDA defect2 operator currently uses float32");
  TORCH_CHECK(input.dim() == 4 && weight.dim() == 4, "input/weight must be 4D");
  TORCH_CHECK(input.is_contiguous() && weight.is_contiguous(), "input/weight must be contiguous");
  int64_t batch = input.size(0);
  int64_t in_channels = input.size(1);
  int64_t height = input.size(2);
  int64_t width = input.size(3);
  int64_t out_channels = weight.size(0);
  int64_t kernel = weight.size(2);
  TORCH_CHECK(weight.size(1) == in_channels && weight.size(3) == kernel, "invalid weight shape");
  TORCH_CHECK(kernel == 1 || kernel == 3 || kernel == 5, "unsupported kernel size");
  TORCH_CHECK(stride == 1 || stride == 2, "unsupported stride");
  TORCH_CHECK(!bias.defined() || bias.numel() == out_channels, "invalid bias shape");
  int64_t padding = kernel / 2;
  int64_t out_height = (height + 2 * padding - kernel) / stride + 1;
  int64_t out_width = (width + 2 * padding - kernel) / stride + 1;
  auto output = torch::empty({batch, out_channels, out_height, out_width}, input.options());
  const c10::cuda::CUDAGuard guard(input.device());
  cudaStream_t stream = at::cuda::getDefaultCUDAStream(input.device().index()).stream();
  int64_t total = output.numel();
  int threads = 256;
  circular_conv_kernel<<<(total + threads - 1) / threads, threads, 0, stream>>>(
      input.data_ptr<float>(), weight.data_ptr<float>(), bias.defined() ? bias.data_ptr<float>() : nullptr,
      output.data_ptr<float>(), batch, in_channels, height, width, out_channels, kernel, stride,
      out_height, out_width, bias.defined() ? 1 : 0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor fused_scale_gate_cuda(
    const torch::Tensor& x,
    const torch::Tensor& coeff,
    const torch::Tensor& local_weight,
    const torch::Tensor& local_bias,
    const torch::Tensor& global_bias,
    const torch::Tensor& matrix,
    double bounded_gain,
    double mix_gain) {
  TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, "CUDA defect2 operator currently uses float32");
  TORCH_CHECK(x.dim() == 4 && coeff.dim() == 4 && local_weight.dim() == 4, "invalid gate tensor rank");
  TORCH_CHECK(x.is_contiguous() && coeff.is_contiguous() && local_weight.is_contiguous(), "gate tensors must be contiguous");
  int64_t batch = x.size(0);
  int64_t channels = x.size(1);
  int64_t height = x.size(2);
  int64_t width = x.size(3);
  int64_t coeff_channels = coeff.size(1);
  TORCH_CHECK(coeff.size(0) == batch && coeff.size(2) == height && coeff.size(3) == width, "invalid coeff shape");
  TORCH_CHECK(local_weight.size(0) == channels && local_weight.size(1) == coeff_channels, "invalid local weight shape");
  TORCH_CHECK(global_bias.sizes() == torch::IntArrayRef({batch, channels}), "invalid global bias shape");
  TORCH_CHECK(!local_bias.defined() || local_bias.numel() == channels, "invalid local bias shape");
  TORCH_CHECK(!matrix.defined() || (matrix.dim() == 3 && matrix.size(0) == batch && matrix.size(1) == channels && matrix.size(2) == channels), "invalid gate matrix shape");
  auto output = torch::empty_like(x);
  const c10::cuda::CUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getDefaultCUDAStream(x.device().index()).stream();
  int64_t total = output.numel();
  int threads = 256;
  scale_gate_kernel<<<(total + threads - 1) / threads, threads, 0, stream>>>(
      x.data_ptr<float>(), coeff.data_ptr<float>(), local_weight.data_ptr<float>(),
      local_bias.defined() ? local_bias.data_ptr<float>() : nullptr, global_bias.data_ptr<float>(),
      matrix.defined() ? matrix.data_ptr<float>() : nullptr, output.data_ptr<float>(), batch, channels,
      coeff_channels, height, width, local_bias.defined() ? 1 : 0, matrix.defined() ? 1 : 0,
      static_cast<float>(bounded_gain), static_cast<float>(mix_gain));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
