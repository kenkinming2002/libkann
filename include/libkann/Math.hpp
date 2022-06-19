#pragma once

#include <libkann/Export.hpp>

#include <libkann/Tensor.hpp>
#include <libkann/Vec.hpp>

namespace kann::math
{
  KANN_EXPORT void add_to(MutableTensorRef to, TensorRef value);

  KANN_EXPORT float norm(TensorRef value);

  KANN_EXPORT Tensor broadcast(Tensor value, Shape shape);
  KANN_EXPORT Tensor reduce(Tensor value, Shape shape);

  /* X = x_1 * ... * x_m
   * Y = y_1 * ... * y_k
   * Z = z_1 * ... * z_n
   *
   * op(a): X * Y
   * op(b): Y * Z
   * output: X * Z */
  KANN_EXPORT Tensor product(Tensor a, Tensor b, size_t rank_m, size_t rank_n, size_t rank_k, bool transpose_a, bool transpose_b);

  /* X = x_1 * ... * x_m
   * Y = y_1 * ... * y_k
   * Z = z_1 * ... * z_n
   *
   * op(input):  X * Y * i_1 * i_2
   * op(kernel): Y * Z * k_1 * k_2
   * output:     X * Z * j_1 * j_2
   *
   * Effect: the same as product with X, Y, Z regarded as tensor of tensor of
   *         rank i_1 * i_2, k_1 * k_2 and j_1 * j_2 respectively, and
   *         multiplication replaced with 2d convolution/cross correlation. */
  KANN_EXPORT Tensor cross_correlate2d(Tensor inputs, Tensor kernels, size_t rank_m, size_t rank_n, size_t rank_k, bool transpose_input, bool transpose_kernel, Vec2 padding_size);
  KANN_EXPORT Tensor convolve2d(Tensor inputs, Tensor kernels, size_t rank_m, size_t rank_n, size_t rank_k, bool transpose_input, bool transpose_kernel, Vec2 padding_size);
}
