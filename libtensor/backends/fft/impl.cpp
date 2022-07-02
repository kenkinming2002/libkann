#include "../Export.hpp"

#include "fft.hpp"

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <complex>

#include <stddef.h>

static inline auto equal(const auto& a, const auto& b)
{
  assert(a == b);
  return a;
}

static inline size_t next_power_of_2(size_t value)
{
  size_t result = 1;
  while(result < value) result <<= 1;
  return result;
}

static inline void _sgecorr2d(Block<float, 4> outputs, Block<const float, 4> inputs, Block<const float, 4> kernels)
{
  const size_t M = equal(outputs.sizes[0], inputs.sizes[0]);
  const size_t N = equal(outputs.sizes[1], kernels.sizes[1]);
  const size_t K = equal(inputs.sizes[1], kernels.sizes[0]);

  const size_t input_height  = inputs.sizes[2],  input_width  = inputs.sizes[3];
  const size_t kernel_height = kernels.sizes[2], kernel_width = kernels.sizes[3];
  const size_t output_height = outputs.sizes[2], output_width = outputs.sizes[3];

  for(size_t m=0; m<M; ++m)
    for(size_t n=0; n<N; ++n)
      for(size_t k=0; k<K; ++k)
      {
        const size_t min_height = input_height + kernel_height - 1;
        const size_t min_width  = input_width  + kernel_width  - 1;

        const size_t height = next_power_of_2(min_height);
        const size_t width  = next_power_of_2(min_width);

        auto input_buf  = std::make_unique_for_overwrite<std::complex<float>[]>(height * width);
        auto kernel_buf = std::make_unique_for_overwrite<std::complex<float>[]>(height * width);

        auto input_block  = make_block(input_buf.get(),  height, width);
        auto kernel_block = make_block(kernel_buf.get(), height, width);

        // Padding
        pad(input_block,  inputs[m][k],  std::complex<float>(0.0f), Direction::RIGHT);
        pad(kernel_block, kernels[k][n], std::complex<float>(0.0f), Direction::LEFT);

        // FFT
        fft(input_block,  Direction::LEFT, Type::FORWARD);
        fft(kernel_block, Direction::RIGHT, Type::BACKWARD);

        // Product
        for(size_t i=0; i<height*width; ++i)
          input_buf[i] *= kernel_buf[i];

        // IFFT
        fft(input_block, Direction::LEFT, Type::BACKWARD);

        // Scale
        for(size_t i=0; i<height*width; ++i)
          input_buf[i] /= height*width;

        auto output_block = input_block.subblock({
          (min_height - output_height) / 2,
          (min_width  - output_width)  / 2},
          {output_height, output_width}
        );
        copy_add(outputs[m][n], output_block, [](std::complex<float> v) { return v.real(); });
      }
}

static inline void _sgeconv2d(Block<float, 4> outputs, Block<const float, 4> inputs, Block<const float, 4> kernels)
{
  const size_t M = equal(outputs.sizes[0], inputs.sizes[0]);
  const size_t N = equal(outputs.sizes[1], kernels.sizes[1]);
  const size_t K = equal(inputs.sizes[1], kernels.sizes[0]);

  const size_t input_height  = inputs.sizes[2],  input_width  = inputs.sizes[3];
  const size_t kernel_height = kernels.sizes[2], kernel_width = kernels.sizes[3];
  const size_t output_height = outputs.sizes[2], output_width = outputs.sizes[3];

  for(size_t m=0; m<M; ++m)
    for(size_t n=0; n<N; ++n)
      for(size_t k=0; k<K; ++k)
      {
        const size_t min_height = input_height + kernel_height - 1;
        const size_t min_width  = input_width  + kernel_width  - 1;

        const size_t height = next_power_of_2(min_height);
        const size_t width  = next_power_of_2(min_width);

        auto input_buf  = std::make_unique_for_overwrite<std::complex<float>[]>(height * width);
        auto kernel_buf = std::make_unique_for_overwrite<std::complex<float>[]>(height * width);

        auto input_block  = make_block(input_buf.get(),  height, width);
        auto kernel_block = make_block(kernel_buf.get(), height, width);

        // Padding
        pad(input_block,  inputs[m][k],  std::complex<float>(0.0f), Direction::RIGHT);
        pad(kernel_block, kernels[k][n], std::complex<float>(0.0f), Direction::RIGHT);

        // FFT
        fft(input_block,  Direction::LEFT, Type::FORWARD);
        fft(kernel_block, Direction::LEFT, Type::FORWARD);

        // Product
        for(size_t i=0; i<height*width; ++i)
          input_buf[i] *= kernel_buf[i];

        // IFFT
        fft(input_block, Direction::LEFT, Type::BACKWARD);

        // Scale
        for(size_t i=0; i<height*width; ++i)
          input_buf[i] /= height*width;

        auto output_block = input_block.subblock({
          (min_height - output_height) / 2,
          (min_width - output_width)   / 2},
          {output_height, output_width}
        );
        copy_add(outputs[m][n], output_block, [](std::complex<float> v) { return v.real(); });
      }
}

extern "C"
{
  BACKEND_EXPORT void sgecorr2d(size_t M, size_t N, size_t K,
      const float* inputs,  size_t input_height,  size_t input_width,  bool trans_inputs,
      const float* kernels, size_t kernel_height, size_t kernel_width, bool trans_kernels,
      float* outputs,       size_t output_height, size_t output_width)
  {
    auto inputs_block = trans_inputs ? make_block(inputs, K, M, input_height, input_width).transpose(0, 1)
                                     : make_block(inputs, M, K, input_height, input_width);

    auto kernels_block = trans_kernels ? make_block(kernels, N, K, kernel_height, kernel_width).transpose(0, 1)
                                       : make_block(kernels, K, N, kernel_height, kernel_width);

    auto outputs_block = make_block(outputs, M, N, output_height, output_width);
    fill(outputs_block, 0.0f);
    _sgecorr2d(outputs_block, inputs_block, kernels_block);
  }

  BACKEND_EXPORT void sgeconv2d(size_t M, size_t N, size_t K,
      const float* inputs,  size_t input_height,  size_t input_width,  bool trans_inputs,
      const float* kernels, size_t kernel_height, size_t kernel_width, bool trans_kernels,
      float* outputs,       size_t output_height, size_t output_width)
  {
    auto inputs_block = trans_inputs ? make_block(inputs, K, M, input_height, input_width).transpose(0, 1)
                                     : make_block(inputs, M, K, input_height, input_width);

    auto kernels_block = trans_kernels ? make_block(kernels, N, K, kernel_height, kernel_width).transpose(0, 1)
                                       : make_block(kernels, K, N, kernel_height, kernel_width);

    auto outputs_block = make_block(outputs, M, N, output_height, output_width);
    fill(outputs_block, 0.0f);
    _sgeconv2d(outputs_block, inputs_block, kernels_block);
  }
}
