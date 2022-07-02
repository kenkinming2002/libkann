#pragma once

#include "types.hpp"
#include "utils.hpp"
#include "block.hpp"

#include <complex>
#include <array>

#include <stddef.h>
#include <assert.h>

// There are two choice of DFT
// Whether, we use
//   Direction::LEFT  => a_0         + a_1x        +... +a_{N-1}x^{N-1}
//   Direction::RIGHT => a_0x^{-N+1} + a_1x^{-N+2} +... +a_{N-1}
// corresponding to whether x to the power 0 is on the left on on the right
//
// For the second case we can multiply by x^N without change the discrete
// fourier transform since all roots of unity evaluate to 1 under x^N, and we
// have instead:
//   Direction::LEFT  => a_0         + a_1x        +... +a_{N-1}x^{N-1}
//   Direction::RIGHT => a_0x        + a_1x^2      +... +a_{N-1}x^N
// corresponding to whether we want to `bias` towards x to the power 0 or N.
//

/********************
 * Declaration Part *
 ********************/
template<size_t N>
inline void fft(Block<std::complex<float>, N> data, Direction direction, Type type);

/***********************
 * Implementation part *
 ***********************/
inline size_t bit_reverse(size_t value, size_t bit_count)
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

inline size_t log2_exact(size_t value)
{
  size_t result = 0;
  while(value >>= 1)
    ++result;
  return result;
}

inline void bit_reverse(std::complex<float> data[], size_t count, size_t stride)
{
  const size_t bit_count = log2_exact(count);
  for(size_t i=0; i<count; ++i)
  {
    size_t j = bit_reverse(i, bit_count);
    if(i < j)
      std::swap(data[i * stride], data[j * stride]);
  }
}

inline void fft1d(std::complex<float> data[], size_t size, size_t stride, Direction direction, Type type)
{
  const size_t depth = log2_exact(size);
  for(size_t k=0; k<depth; ++k)
  {
    const size_t chunk_count     = 1 << k;
    const size_t chunk_size      = size >> k;
    const size_t chunk_half_size = chunk_size >> 1;

    for(size_t i=0; i<chunk_count; ++i)
      for(size_t j=0; j<chunk_half_size; ++j)
      {
        const float sgn   = type == Type::FORWARD  ? -1.0f
                          : type == Type::BACKWARD ? 1.0f
                          : (assert(false && "Unimplemented"), 0.0f);

        const float ratio = direction == Direction::LEFT  ? float(j)   / float(chunk_size)
                          : direction == Direction::RIGHT ? float(j+1) / float(chunk_size)
                          : (assert(false && "Unimplemented"), 0.0f);

        const std::complex<float> omega = std::polar(1.0f, 2.0f * sgn * std::numbers::pi_v<float> * ratio);

        const size_t left_index  = (i * chunk_size +                   j) * stride;
        const size_t right_index = (i * chunk_size + chunk_half_size + j) * stride;

        const std::complex<float> a = data[left_index];
        const std::complex<float> b = data[right_index];

        data[left_index]  = a + b;
        data[right_index] = (a - b) * omega;
      }
  }
  bit_reverse(data, size, stride);
}

template<size_t N>
inline void _fft(Block<std::complex<float>, N> data, Direction direction, Type type)
{
  for(size_t i=0; i<data.sizes[0]; ++i)
    _fft(data[i], direction, type);
}

template<>
inline void _fft(Block<std::complex<float>, 1> data, Direction direction, Type type)
{
  fft1d(data.data, data.sizes[0], data.strides[0], direction, type);
}

template<size_t N>
inline void fft(Block<std::complex<float>, N> data, Direction direction, Type type)
{
  for(size_t i=0; i<N; ++i)
    _fft(data.move_last(i), direction, type);
}

