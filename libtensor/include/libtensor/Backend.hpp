#pragma once

#include <vector>
#include <stddef.h>

namespace tensor
{
  // Single precision general matrix multiply
  typedef void(*sgemm_t)(size_t M, size_t N, size_t K,
      const float* A, bool trans_a,
      const float* B, bool trans_b,
      float* C);

  // Single precision general cross-correlation 2D
  typedef void(*sgecorr2d_t)(size_t M, size_t N, size_t K,
      const float* inputs,  size_t height_input, size_t width_input, bool trans_inputs,
      const float* kernels, size_t height_kernel, size_t width_kernel, bool trans_kernels,
      float* outputs, size_t height_output, size_t width_output);

  // Single precision general convolution 2D
  typedef void(*sgeconv2d_t)(size_t M, size_t N, size_t K,
      const float* inputs,  size_t height_input, size_t width_input, bool trans_inputs,
      const float* kernels, size_t height_kernel, size_t width_kernel, bool trans_kernels,
      float* outputs, size_t height_output, size_t width_output);

  struct Backend
  {
    sgemm_t sgemm;
    sgecorr2d_t sgecorr2d;
    sgeconv2d_t sgeconv2d;
  };

  std::vector<Backend> backends();
}
