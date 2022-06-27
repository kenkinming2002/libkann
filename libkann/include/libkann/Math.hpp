#pragma once

#include <libkann/Export.hpp>

#include <libkann/Tensor.hpp>
#include <libkann/Vec.hpp>

namespace kann::math
{
  KANN_EXPORT float norm(const Tensor<float>& value);

  enum class Direction { LEFT, RIGHT };

  template<size_t K> void broadcast(Tensor<float> dst, std::array<Tensor<const float>, K> srcs, Direction direction, const auto& impl);
  template<size_t K> void reduce   (Tensor<float> dst, std::array<Tensor<const float>, K> srcs, Direction direction, const auto& impl);
  template<size_t K> void transform(Tensor<float> dst, std::array<Tensor<const float>, K> srcs, const auto& impl);

  static constexpr auto FMA(float value)  { return [=](float output, float input) { return output + value * input;  }; }
  static constexpr auto SCALE(float value)  { return [=](float output, float input) { return value * input;  }; }

  static constexpr auto ADD  = [](float output, auto... inputs) { return output + (inputs + ...);  };
  static constexpr auto MUL  = [](float output, auto... inputs) { return output * (inputs * ...);  };

  static constexpr auto SUB  = [](float output, float input) { return output - input;  };
  static constexpr auto DIV  = [](float output, float input) { return output / input;  };

  KANN_EXPORT void product(Tensor<float> dst,
      Tensor<const float> a, bool transpose_a,
      Tensor<const float> b, bool transpose_b);

  enum class Image2DOperation { CROSS_CORRELATION, CONVOLUTION };
  KANN_EXPORT void image2d_operation(Tensor<float> outputs,
      Tensor<const float> inputs, bool transpose_inputs,
      Tensor<const float> kernels, bool transpose_kernels,
      Image2DOperation operation);

  /******************
   * Implementation *
   ******************/
  template<size_t K>
  void broadcast(Tensor<float> dst, std::array<Tensor<const float>, K> srcs, Direction direction, const auto& impl) requires(K != 0)
  {
    assert(dst.rank() == 2);
    assert(ranges::all_of(srcs, [](const auto& src) { return src.rank() == 1; }));

    const auto& [M, N] = std::make_pair(dst.dimension(0), dst.dimension(1));
    if(direction == Direction::LEFT)
    {
      assert(ranges::all_of(srcs, [N=N](const auto& src) { return src.dimension(0) == N; }));
      for(size_t i=0; i<M; ++i)
        for(size_t j=0; j<N; ++j)
        {
          std::array<float, K> inputs;
          for(size_t k=0; k<K; ++k)
            inputs[k] = srcs[k](j);

          float& output = dst(i,j);
          std::apply([&](auto... inputs) { output = impl(output, inputs...); }, inputs);
        }
    }
    else if(direction == Direction::RIGHT)
    {
      assert(ranges::all_of(srcs, [M=M](const auto& src) { return src.dimension(0) == M; }));
      for(size_t i=0; i<M; ++i)
        for(size_t j=0; j<N; ++j)
        {
          std::array<float, K> inputs;
          for(size_t k=0; k<K; ++k)
            inputs[k] = srcs[k](i);

          float& output = dst(i,j);
          std::apply([&](auto... inputs) { output = impl(output, inputs...); }, inputs);
        }
    }
    else
      assert(false && "Unreachable");
  }

  template<size_t K>
  void reduce(Tensor<float> dst, std::array<Tensor<const float>, K> srcs, Direction direction, const auto& impl) requires(K != 0)
  {
    assert(dst.rank() == 1);
    assert(ranges::all_of(srcs, [](const auto& src) { return src.rank() == 2; }));

    const auto& [M, N] = std::make_pair(srcs.front().dimension(0), srcs.front().dimension(1));
    assert(ranges::all_of(srcs, [M=M, N=N](const auto& src) { return src.dimension(0) == M && src.dimension(1) == N; }));
    if(direction == Direction::LEFT)
    {
      assert(dst.dimension(0) == N);
      for(size_t i=0; i<M; ++i)
        for(size_t j=0; j<N; ++j)
        {
          std::array<float, K> inputs;
          for(size_t k=0; k<K; ++k)
            inputs[k] = srcs[k](i, j);

          float& output = dst(j);
          std::apply([&](auto... inputs) { output = impl(output, inputs...); }, inputs);
        }
    }
    else if(direction == Direction::RIGHT)
    {
      assert(dst.dimension(0) == M);
      for(size_t i=0; i<M; ++i)
        for(size_t j=0; j<N; ++j)
        {
          std::array<float, K> inputs;
          for(size_t k=0; k<K; ++k)
            inputs[k] = srcs[k](i, j);

          float& output = dst(i);
          std::apply([&](auto... inputs) { output = impl(output, inputs...); }, inputs);
        }
    }
    else
      assert(false && "Unreachable");
  }

  template<size_t K>
  void transform(Tensor<float> dst, std::array<Tensor<const float>, K> srcs, const auto& impl)
  {
    assert(dst.rank() == 1);
    assert(ranges::all_of(srcs, [](const auto& src) { return src.rank() == 1; }));

    const size_t N = dst.size();
    assert(ranges::all_of(srcs, [&](const auto& src) { return src.dimension(0) == N; }));
    for(size_t i=0; i<N; ++i)
    {
      std::array<float, K> inputs;
      for(size_t k=0; k<K; ++k)
        inputs[k] = srcs[k](i);

      float& output = dst(i);
      std::apply([&](auto... inputs) { output = impl(output, inputs...); }, inputs);
    }
  }
}
