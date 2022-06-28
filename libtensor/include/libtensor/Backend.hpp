#pragma once

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
      const float* A, size_t height_A, size_t width_A, bool trans_a,
      const float* B, size_t height_B, size_t width_B, bool trans_b,
      float* C);

  // Single precision general convolution 2D
  typedef void(*sgeconv2d_t)(size_t M, size_t N, size_t K,
      const float* A, size_t height_A, size_t width_A, bool trans_a,
      const float* B, size_t height_B, size_t width_B, bool trans_b,
      float* C);

  struct Backend
  {
    sgemm_t sgemm;
    sgecorr2d_t sgecorr2d;
    sgeconv2d_t sgeconv2d;
  };

  std::vector<Backend> backends();
}
