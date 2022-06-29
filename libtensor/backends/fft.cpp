#include "Export.hpp"

#include <memory>
#include <complex>

#include <stddef.h>
#include <assert.h>

/****************
 * FFT and IFFT *
 ****************/
static inline void fft(size_t N, const std::complex<float> inputs[],  size_t input_stride,
                                       std::complex<float> outputs[], size_t output_stride)
{
  if(N == 1)
  {
    outputs[0] = inputs[0];
    return;
  }

  // We interleave inputs but split outputs in the middle into the recusive calls
  fft(N/2, inputs               , input_stride*2, outputs                      , output_stride);
  fft(N/2, inputs + input_stride, input_stride*2, outputs + N/2 * output_stride, output_stride);

  // Collection step
  for(size_t i=0; i<N/2; ++i)
  {
    // TODO: Precompute table
    const std::complex<float> omega_i = std::polar(1.0f, -2.0f * std::numbers::pi_v<float> * ((float)i / N));

    const std::complex<float> y_e = outputs[i           * output_stride];
    const std::complex<float> y_o = outputs[(i + N / 2) * output_stride];
    outputs[i           * output_stride] = y_e + omega_i * y_o;
    outputs[(i + N / 2) * output_stride] = y_e - omega_i * y_o;
  }
}

static inline void _ifft(size_t N, const std::complex<float> inputs[],  size_t input_stride,
                                         std::complex<float> outputs[], size_t output_stride)
{
  if(N == 1)
  {
    outputs[0] = inputs[0];
    return;
  }

  // We interleave inputs but split outputs in the middle into the recusive calls
  _ifft(N/2, inputs               , input_stride*2, outputs                      , output_stride);
  _ifft(N/2, inputs + input_stride, input_stride*2, outputs + N/2 * output_stride, output_stride);

  // Collection step
  for(size_t i=0; i<N/2; ++i)
  {
    // TODO: Precompute table
    const std::complex<float> omega_i = std::polar(1.0f, 2.0f * std::numbers::pi_v<float> * ((float)i / N));

    const std::complex<float> y_e = outputs[i           * output_stride];
    const std::complex<float> y_o = outputs[(i + N / 2) * output_stride];
    outputs[i           * output_stride] = y_e + omega_i * y_o;
    outputs[(i + N / 2) * output_stride] = y_e - omega_i * y_o;
  }
}

static inline void ifft(size_t N, const std::complex<float> inputs[],  size_t input_stride,
                                        std::complex<float> outputs[], size_t output_stride)
{
  _ifft(N, inputs, input_stride, outputs, output_stride);
  for(size_t i=0; i<N; ++i)
    outputs[i * output_stride] /= N;
}

/*******************
 * 2D FFT and IFFT *
 *******************/
static inline void fft2d(size_t height, size_t width, std::complex<float> values[], std::complex<float> tmp[])
{
  // FFT on each row
  for(size_t y=0; y<height; ++y)
    fft(width, &values[y*width], 1, &tmp[y*width], 1);

  // FFT on each column
  for(size_t x=0; x<width; ++x)
    fft(height, &tmp[x], width, &values[x], width);
}

static inline void ifft2d(size_t height, size_t width, std::complex<float> values[], std::complex<float> tmp[])
{
  // IFFT on each row
  for(size_t y=0; y<height; ++y)
    ifft(width, &values[y*width], 1, &tmp[y*width], 1);

  // IFFT on each column
  for(size_t x=0; x<width; ++x)
    ifft(height, &tmp[x], width, &values[x], width);
}

/***********
 * Padding *
 ***********/
static inline void pad2d(size_t input_height,  size_t input_width,  const float input[],
                         size_t output_height, size_t output_width, std::complex<float> output[])
{
  for(size_t y=0; y<output_height; ++y)
    for(size_t x=0; x<output_width; ++x)
      if(y < input_height && x < input_width)
        output[y * output_width + x] = input[y * input_width + x];
      else
        output[y * output_width + x] = 0.0f;
}

static inline void unpad2d(size_t input_height,   size_t input_width,  const std::complex<float> input[],
                           size_t output_height,  size_t output_width, float output[],
                           size_t offset_y,       size_t offset_x)
{
  // If we don't do any offset we would the result of fully padded convolution
  for(size_t y=0; y<output_height; ++y)
    for(size_t x=0; x<output_width; ++x)
      output[y * output_width + x] += input[(offset_y + y) * input_width + (offset_x + x)].real();
}

/**************
 * Power of 2 *
 **************/
static inline size_t next_power_of_2(size_t value)
{
  size_t result = 1;
  while(result < value) result <<= 1;
  return result;
}

/**********
 * Single *
 **********/
void sconv2d(size_t input_height,  size_t input_width,  const float input[],
             size_t kernel_height, size_t kernel_width, const float kernel[],
             size_t output_height, size_t output_width, float output[])
{
  // This is also the maximum output_height and output_width
  const size_t min_height = input_height + kernel_height - 1;
  const size_t min_width  = input_width  + kernel_width  - 1; //

  // Round up to next power of two
  const size_t height = next_power_of_2(min_height);
  const size_t width  = next_power_of_2(min_width);

  auto input_buf  = std::make_unique_for_overwrite<std::complex<float>[]>(height * width);
  auto kernel_buf = std::make_unique_for_overwrite<std::complex<float>[]>(height * width);
  auto tmp        = std::make_unique_for_overwrite<std::complex<float>[]>(height * width);

  // Pad and FFT
  pad2d(input_height, input_width, input, height, width, input_buf.get());
  fft2d(height, width, input_buf.get(), tmp.get());

  pad2d(kernel_height, kernel_width, kernel, height, width, kernel_buf.get());
  fft2d(height, width, kernel_buf.get(), tmp.get());

  // Multiply input_buf and kernel_buf together
  for(size_t i=0; i<height*width; ++i)
    input_buf[i] *= kernel_buf[i];

  // IFFT and unpad
  ifft2d(height, width, input_buf.get(), tmp.get());
  unpad2d(height, width, input_buf.get(), output_height, output_width, output,
      (min_height - output_height) / 2,
      (min_width  - output_width)  / 2);
}

extern "C"
{
  BACKEND_EXPORT void sgeconv2d(size_t M, size_t N, size_t K,
      const float* inputs,  size_t input_height,  size_t input_width,  bool trans_inputs,
      const float* kernels, size_t kernel_height, size_t kernel_width, bool trans_kernels,
      float* outputs,       size_t output_height, size_t output_width)
  {
    std::fill_n(outputs, M * N * output_height * output_width, 0.0f);
    for(size_t m=0; m<M; ++m)
      for(size_t n=0; n<N; ++n)
        for(size_t k=0; k<K; ++k)
        {
          const float* input  = &inputs [input_height  * input_width  * ( trans_inputs  ? (k * M + m) : (m * K + k) )];
          const float* kernel = &kernels[kernel_height * kernel_width * ( trans_kernels ? (n * K + k) : (k * N + n) )];
          float* output = &outputs[output_height * output_width * (m * N + n)];
          sconv2d(input_height, input_width, input, kernel_height, kernel_width, kernel, output_height, output_width, output);
        }
  }
}
