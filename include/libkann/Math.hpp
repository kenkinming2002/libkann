#pragma once

#include <libkann/Export.hpp>

#include <libkann/Tensor.hpp>
#include <libkann/Vec.hpp>

namespace kann::math
{
  KANN_EXPORT float norm(TensorRef value);

  enum class Direction { LEFT, RIGHT };
  enum class Operation { STORE, ADD, SUB, MUL, DIV };
  KANN_EXPORT void broadcast(TensorRef value, MutableTensorRef target, Operation operation, Direction direction);
  KANN_EXPORT void reduce(TensorRef value, MutableTensorRef target, Operation operation, Direction direction);

  KANN_EXPORT void product(TensorRef a, bool tranpose_a, TensorRef b, bool transpose_b, MutableTensorRef c);

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

  template<typename Impl>
  Tensor cwise(Tensor a, Impl impl)
  {
    const Shape& shape = a.shape();
    const size_t size = a.size();
    MutableTensor result = MutableTensor::create(shape);
    for(size_t i=0; i<size; ++i)
      result.get(i) = impl(a.get(i));

    return std::move(result).as_const();
  }

  template<typename Impl>
  Tensor cwise(Tensor a, Tensor b, Impl impl)
  {
    assert(a.shape() == b.shape());
    const Shape& shape = a.shape();
    const size_t size = a.size();
    MutableTensor result = MutableTensor::create(shape);
    for(size_t i=0; i<size; ++i)
      result.get(i) = impl(a.get(i), b.get(i));

    return std::move(result).as_const();
  }

  template<typename Impl>
  Tensor cwise(Tensor a, Tensor b, Tensor c, Impl impl)
  {
    assert(a.shape() == b.shape() && b.shape() == c.shape());
    const Shape& shape = a.shape();
    const size_t size = a.size();
    MutableTensor result = MutableTensor::create(shape);
    for(size_t i=0; i<size; ++i)
      result.get(i) = impl(a.get(i), b.get(i), c.get(i));

    return std::move(result).as_const();
  }

  KANN_EXPORT Tensor add(Tensor a, Tensor b);
  KANN_EXPORT Tensor sub(Tensor a, Tensor b);
  KANN_EXPORT Tensor scale(Tensor a, float factor);
}
