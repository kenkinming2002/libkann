#include "Export.hpp"

#include <vector>
#include <memory>
#include <complex>

#include <stddef.h>
#include <assert.h>

// Inplace non-recusvie implementation of FFT and IFFT
// Refer to : http://wwwa.pikara.ne.jp/okojisan/otfft-en/cooley-tukey.html for more details
template<typename T>
struct slice
{
public:
  T* data;
  size_t offset;
  size_t stride;
  size_t count;

public:
  constexpr slice(T* data, size_t count, size_t offset = 0, size_t stride = 1)
    : data(data), offset(offset), stride(stride), count(count) {}

  constexpr slice(std::vector<T>& data)
    : slice(data.data(), data.size()) {}

public:
  const T& operator[](size_t index) const
  {
    assert(index < count);
    return data[offset + index * stride];
  }

  T& operator[](size_t index)
  {
    assert(index < count);
    return data[offset + index * stride];
  }
};

static inline size_t log2_exact(size_t value)
{
  size_t result = 0;
  while(value >>= 1)
    ++result;
  return result;
}

static inline size_t bit_reverse(size_t value, size_t bit_count)
{
  size_t result = 0;
  for(size_t i=0; i<bit_count; ++i)
  {
    size_t bit = value & 0x1;
    value >>= 1;
    result <<= 1;
    result = result | bit;
  }
  return result;
}

static inline void bit_reverse(slice<std::complex<float>> data)
{
  const size_t N = data.count;
  const size_t bit_count = log2_exact(N);
  for(size_t i=0; i<N; ++i)
  {
    size_t j = bit_reverse(i, bit_count);
    if(i < j)
      std::swap(data[i], data[j]);
  }
}

static inline void fft(slice<std::complex<float>> data)
{
  const size_t N = data.count;
  const size_t depth = log2_exact(N);
  for(size_t k=0; k<depth; ++k)
  {
    const size_t chunk_count     = 1 << k;
    const size_t chunk_size      = N >> k;
    const size_t chunk_half_size = chunk_size >> 1;

    for(size_t i=0; i<chunk_count; ++i)
      for(size_t j=0; j<chunk_half_size; ++j)
      {
        // Operate on i * chunk_size +jk and i * chunk_size + chunk_half_size + j
        const std::complex<float> a = data[i * chunk_size +                   j];
        const std::complex<float> b = data[i * chunk_size + chunk_half_size + j];
        const std::complex<float> omega = std::polar(1.0f, -2.0f * std::numbers::pi_v<float> * (float)j / (float)chunk_size);
        data[i * chunk_size +                   j] = a + b;
        data[i * chunk_size + chunk_half_size + j] = (a - b) * omega;
      }
  }
  bit_reverse(data);
}


static inline void ifft(slice<std::complex<float>> data)
{
  const size_t N = data.count;
  const size_t depth = log2_exact(N);
  for(size_t k=0; k<depth; ++k)
  {
    const size_t chunk_count     = 1 << k;
    const size_t chunk_size      = N >> k;
    const size_t chunk_half_size = chunk_size >> 1;

    for(size_t i=0; i<chunk_count; ++i)
      for(size_t j=0; j<chunk_half_size; ++j)
      {
        // Operate on i * chunk_size +jk and i * chunk_size + chunk_half_size + j
        const std::complex<float> a = data[i * chunk_size +                   j];
        const std::complex<float> b = data[i * chunk_size + chunk_half_size + j];
        const std::complex<float> omega = std::polar(1.0f, 2.0f * std::numbers::pi_v<float> * (float)j / (float)chunk_size);
        data[i * chunk_size +                   j] = a + b;
        data[i * chunk_size + chunk_half_size + j] = (a - b) * omega;
      }
  }
  bit_reverse(data);
  for(size_t i=0; i<N; ++i)
    data[i] /= N;
}

/*******************
 * 2D FFT and IFFT *
 *******************/
static inline void fft2d(size_t height, size_t width, std::complex<float> values[])
{
  // FFT on each row
  for(size_t y=0; y<height; ++y)
    fft(slice(values, width, y * width, 1));

  // FFT on each column
  for(size_t x=0; x<width; ++x)
    fft(slice(values, height, x, width));
}

static inline void ifft2d(size_t height, size_t width, std::complex<float> values[])
{
  // IFFT on each row
  for(size_t y=0; y<height; ++y)
    ifft(slice(values, width, y * width, 1));

  // IFFT on each column
  for(size_t x=0; x<width; ++x)
    ifft(slice(values, height, x, width));
}

/***********
 * Padding *
 ***********/
static inline void pad2d_reverse(size_t input_height,  size_t input_width,  const float input[],
                         size_t output_height, size_t output_width, std::complex<float> output[])
{
  for(size_t y=0; y<output_height; ++y)
    for(size_t x=0; x<output_width; ++x)
      if(y < input_height && x < input_width)
        output[y * output_width + x] = input[(input_height - y - 1) * input_width + (input_width - x - 1)];
      else
        output[y * output_width + x] = 0.0f;
}

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
void scorr2d(size_t input_height,  size_t input_width,  const float input[],
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

  // Pad and FFT
  pad2d(input_height, input_width, input, height, width, input_buf.get());
  fft2d(height, width, input_buf.get());

  pad2d_reverse(kernel_height, kernel_width, kernel, height, width, kernel_buf.get());
  fft2d(height, width, kernel_buf.get());

  // Multiply input_buf and kernel_buf together
  for(size_t i=0; i<height*width; ++i)
    input_buf[i] *= kernel_buf[i];

  // IFFT and unpad
  ifft2d(height, width, input_buf.get());
  unpad2d(height, width, input_buf.get(), output_height, output_width, output,
      (min_height - output_height) / 2,
      (min_width  - output_width)  / 2);
}

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

  // Pad and FFT
  pad2d(input_height, input_width, input, height, width, input_buf.get());
  fft2d(height, width, input_buf.get());

  pad2d(kernel_height, kernel_width, kernel, height, width, kernel_buf.get());
  fft2d(height, width, kernel_buf.get());

  // Multiply input_buf and kernel_buf together
  for(size_t i=0; i<height*width; ++i)
    input_buf[i] *= kernel_buf[i];

  // IFFT and unpad
  ifft2d(height, width, input_buf.get());
  unpad2d(height, width, input_buf.get(), output_height, output_width, output,
      (min_height - output_height) / 2,
      (min_width  - output_width)  / 2);
}

extern "C"
{
  BACKEND_EXPORT void sgecorr2d(size_t M, size_t N, size_t K,
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
          scorr2d(input_height, input_width, input, kernel_height, kernel_width, kernel, output_height, output_width, output);
        }
  }

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
