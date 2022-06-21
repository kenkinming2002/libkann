#pragma once

#include <libkann/Export.hpp>

#include <libkann/Tensor.hpp>
#include <libkann/Vec.hpp>

namespace kann::math
{
  KANN_EXPORT float norm(TensorRef value);

  enum class Direction { LEFT, RIGHT };

  struct Operation
  {
  public:
    virtual void process(const float& from, float& to) const = 0;
    virtual ~Operation() = default;

  public:
    template<typename Impl>
    static constexpr auto create(Impl impl)
    {
      struct OperationImpl final : public Operation
      {
      public:
        constexpr OperationImpl(Impl impl) : impl(impl) {}

      public:
        Impl impl;
        void process(const float& from, float& to) const override { impl(from, to); }
      };
      return OperationImpl(impl);
    }
  };

  inline auto STORE = Operation::create([](const float& from, float& to) { to  = from; });
  inline auto ADD   = Operation::create([](const float& from, float& to) { to += from; });
  inline auto SUB   = Operation::create([](const float& from, float& to) { to -= from; });
  inline auto MUL   = Operation::create([](const float& from, float& to) { to *= from; });
  inline auto DIV   = Operation::create([](const float& from, float& to) { to /= from; });
  inline auto FMA(float value) { return Operation::create([=](const float& from, float& to) { to = to + value * from; }); }

  KANN_EXPORT void broadcast(TensorRef from, MutableTensorRef to, Direction direction, const Operation& operation);
  KANN_EXPORT void reduce(TensorRef from, MutableTensorRef to, Direction direction, const Operation& operation);
  KANN_EXPORT void transform(TensorRef from, MutableTensorRef to, const Operation& operation);

  struct BinaryOperation
  {
  public:
    virtual void process(const float& from1, const float& from2, float& to) const = 0;
    virtual ~BinaryOperation() = default;

  public:
    template<typename Impl>
    static constexpr auto create(Impl impl)
    {
      struct OperationImpl final : public BinaryOperation
      {
      public:
        constexpr OperationImpl(Impl impl) : impl(impl) {}

      public:
        Impl impl;
        void process(const float& from1, const float& from2, float& to) const override { impl(from1, from2, to); }
      };
      return OperationImpl(impl);
    }
  };

  KANN_EXPORT void broadcast2(TensorRef from1, TensorRef from2, MutableTensorRef to, Direction direction, const BinaryOperation& operation);
  KANN_EXPORT void reduce2(TensorRef from1, TensorRef from2,  MutableTensorRef to, Direction direction, const BinaryOperation& operation);
  KANN_EXPORT void transform2(TensorRef from1, TensorRef from2, MutableTensorRef to, const BinaryOperation& operation);

  template<typename Func>
  void transform(TensorRef a, TensorRef b, MutableTensorRef to, Func func)
  {
    assert(a.shape() == to.shape());
    assert(b.shape() == to.shape());
    for(size_t i=0; i<to.size(); ++i)
      to.get(i) = func(a.get(i), b.get(i));
  }

  template<typename Func>
  void transform(TensorRef a, TensorRef b, TensorRef c, MutableTensorRef to, Func func)
  {
    assert(a.shape() == to.shape());
    assert(b.shape() == to.shape());
    assert(c.shape() == to.shape());
    for(size_t i=0; i<to.size(); ++i)
      to.get(i) = func(a.get(i), b.get(i), c.get(i));
  }

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
}
