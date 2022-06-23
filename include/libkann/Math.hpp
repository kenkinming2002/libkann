#pragma once

#include <libkann/Export.hpp>

#include <libkann/Tensor.hpp>
#include <libkann/Vec.hpp>

namespace kann::math
{
  KANN_EXPORT float norm(TensorRef<const float> value);

  enum class Direction { LEFT, RIGHT };

  template<size_t N> void broadcast(TensorRef<float> dst, std::array<TensorRef<const float>, N> srcs, Direction direction, const auto& impl);
  template<size_t N> void reduce   (TensorRef<float> dst, std::array<TensorRef<const float>, N> srcs, Direction direction, const auto& impl);
  template<size_t N> void transform(TensorRef<float> dst, std::array<TensorRef<const float>, N> srcs, const auto& impl);

  static constexpr auto FMA(float value)  { return [=](float output, float input) { return output + value * input;  }; }
  static constexpr auto SCALE(float value)  { return [=](float output, float input) { return value * input;  }; }

  static constexpr auto ADD  = [](float output, auto... inputs) { return output + (inputs + ...);  };
  static constexpr auto MUL  = [](float output, auto... inputs) { return output * (inputs * ...);  };

  static constexpr auto SUB  = [](float output, float input) { return output - input;  };
  static constexpr auto DIV  = [](float output, float input) { return output / input;  };

  KANN_EXPORT void product(MutableTensorRef dst, TensorRef a, bool transpose_a, TensorRef b, bool transpose_b);

  enum class Image2DOperation { CROSS_CORRELATION, CONVOLUTION };
  KANN_EXPORT void image2d_operation(MutableTensorRef outputs, TensorRef inputs, bool transpose_inputs, TensorRef kernels, bool transpose_kernels, Image2DOperation operation);

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

  /******************
   * Implementation *
   ******************/
  namespace details
  {
    inline constexpr auto split_by(Shape a, Shape b, Direction direction)
    {
      switch(direction)
      {
      case Direction::LEFT: // Pad to the left
        assert(a.back(b.rank()) == b);
        return std::make_pair(a.drop_back(b.rank()), b);
      case Direction::RIGHT: // Pad to the right
        assert(a.front(b.rank()) == b);
        return std::make_pair(b, a.drop_front(b.rank()));
      default:
        assert(false && "Unreachable");
      }
    };

    template<typename From, size_t N>
    inline constexpr auto array_transform(std::array<From, N>& values, const auto& f)
    {
      return [&]<size_t... Is>(std::index_sequence<Is...>) {
        return std::array{f(values[Is])...};
      }(std::make_index_sequence<N>());
    }
  }

  template<size_t N>
  void broadcast(TensorRef<float> dst, std::array<TensorRef<const float>, N> srcs, Direction direction, const auto& impl)
  {
    // Step 1: Compute shape
    assert(!srcs.empty());
    const auto& [left, right] = details::split_by(dst.shape(), srcs.front().shape(), direction);

    // Step 2: Reshape
    dst = dst.reshape(Shape(left.size(), right.size()));
    for(TensorRef<const float>& src : srcs)
      src = src.reshape(Shape(src.size()));


    // Step 3: Compute
    for(size_t i=0; i<left.size(); ++i)
      for(size_t j=0; j<right.size(); ++j)
      {
        float* output = &dst[i][j].as_scalar();

        std::array<float, N> inputs;
        switch(direction)
        {
        case Direction::LEFT:  inputs = details::array_transform(srcs, [&](auto& src) { return src[j].as_scalar(); }); break;
        case Direction::RIGHT: inputs = details::array_transform(srcs, [&](auto& src) { return src[i].as_scalar(); }); break;
        }

        std::apply([&](auto... inputs) { *output = impl(*output, inputs...); }, inputs);
      }
  }

  template<size_t N>
  void reduce(TensorRef<float> dst, std::array<TensorRef<const float>, N> srcs, Direction direction, const auto& impl)
  {
    // Step 1: Compute shape
    assert(!srcs.empty());
    const auto& [left, right] = details::split_by(srcs.front().shape(), dst.shape(), direction);

    // Step 2: Reshape
    dst = dst.flatten();
    for(TensorRef<const float>& src : srcs)
      src = src.reshape(Shape(left.size(), right.size()));


    // Step 3: Compute
    for(size_t i=0; i<left.size(); ++i)
      for(size_t j=0; j<right.size(); ++j)
      {
        float* output;
        switch(direction)
        {
        case Direction::LEFT:   output = &dst[j].as_scalar(); break;
        case Direction::RIGHT:  output = &dst[i].as_scalar(); break;
        }

        std::array<float, N> inputs = details::array_transform(srcs, [&](auto& src) { return src[i][j].as_scalar(0); });

        std::apply([&](auto... inputs) { *output = impl(*output, inputs...); }, inputs);
      }
  }

  template<size_t N>
  void transform(TensorRef<float> dst, std::array<TensorRef<const float>, N> srcs, const auto& impl)
  {
    for(TensorRef<const float>& src : srcs)
      assert(dst.shape() == src.shape());

    dst = dst.flatten();
    for(TensorRef<const float>& src : srcs)
      src = src.reshape(Shape(src.size()));

    for(size_t i=0; i<dst.size(); ++i)
    {
      float* output = &dst[i].as_scalar();
      std::array<float, N> inputs = details::array_transform(srcs, [&](auto& src) { return src[i].as_scalar(); });
      std::apply([&](auto... inputs) { *output = impl(*output, inputs...); }, inputs);
    }
  }
}
