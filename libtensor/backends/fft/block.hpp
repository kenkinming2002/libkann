#pragma once

#include "types.hpp"
#include "utils.hpp"

#include <tuple>
#include <array>

#include <stddef.h>
#include <assert.h>

// TODO: We may want to include this in the main libtensor library since it is
//       so much more convenient compare to the shape and math API
template<typename T, size_t N>
struct Block
{
public:
  T* data;

  std::array<size_t, N> strides;
  std::array<size_t, N> sizes;

public:
  operator Block<const T, N>() const
  {
    return Block<const T, N>{
      .data    = this->data,
      .strides = this->strides,
      .sizes   = this->sizes
    };
  }

public:
  constexpr Block<T, N-1> operator[](size_t i) const requires(N>0)
  {
    const size_t stride = this->strides[0];
    const size_t size   = this->sizes[0];

    assert(i<size);
    return Block<T, N-1>{
      .data    = this->data + i * stride,
      .strides = tail(this->strides),
      .sizes   = tail(this->sizes)
    };
  }

  constexpr Block subblock(std::array<size_t, N> offsets, std::array<size_t, N> sizes) const
  {
    size_t offset = 0;
    for(size_t i=0; i<N; ++i)
    {
      assert(offsets[i] + sizes[i] <= this->sizes[i]);
      offset += offsets[i] * this->strides[i];
    }

    return Block{
      .data    = this->data + offset,
      .strides = this->strides,
      .sizes   = sizes
    };
  }

  constexpr Block transpose(size_t i, size_t j)
  {
    assert(i < N);
    assert(j < N);

    Block result = *this;
    std::swap(result.strides[i], result.strides[j]);
    std::swap(result.sizes  [i], result.sizes  [j]);
    return result;
  }

  // Move dimension n to the last dimension
  constexpr Block move_last(size_t n) const
  {
    assert(n < N);

    Block result = *this;
    for(size_t i=n; i<N-1; ++i)
    {
      result.strides[i] = this->strides[i+1];
      result.sizes  [i] = this->sizes  [i+1];
    }
    result.strides[N-1] = this->strides[n];
    result.sizes  [N-1] = this->sizes  [n];
    return result;
  }
};

// make_block overload
// TODO: Make this generic over all dimension
template<typename T>
Block<T, 1> make_block(T* data, size_t size)
{
  return Block<T, 1>{
    .data = data,
    .strides = { 1 },
    .sizes = { size }
  };
}

template<typename T>
Block<T, 2> make_block(T* data, size_t size1, size_t size2)
{
  return Block<T, 2>{
    .data = data,
    .strides = { size2, 1},
    .sizes = { size1, size2 }
  };
}

template<typename T>
Block<T, 3> make_block(T* data, size_t size1, size_t size2, size_t size3)
{
  return Block<T, 3>{
    .data = data,
    .strides = { size2 * size3, size3, 1},
    .sizes = { size1, size2, size3 }
  };
}

template<typename T>
Block<T, 4> make_block(T* data, size_t size1, size_t size2, size_t size3, size_t size4)
{
  return Block<T, 4>{
    .data = data,
    .strides = { size2 * size3 * size4, size3 * size4, size4, 1},
    .sizes = { size1, size2, size3, size4 }
  };
}

template<typename T, size_t N>
Block<T, 1> make_block(std::array<T, N>& data)
{
  return Block<T, 1>{
    .data = data.data(),
    .strides = { 1 },
    .sizes = { data.size() }
  };
}

template<typename T, size_t N>
Block<const T, 1> make_block(const std::array<T, N>& data)
{
  return Block<const T, 1>{
    .data = data.data(),
    .strides = { 1 },
    .sizes = { data.size() }
  };
}

template<typename T, size_t N>
void fill(Block<T, N> block, const T& value)
{
  if constexpr(N>0)
  {
    for(size_t i=0; i<block.sizes[0]; ++i)
      fill(block[i], value);
  }
  else
    block.data[0] = value;
}

template<typename T, typename U, size_t N, typename F>
void copy(Block<T, N> output, Block<U, N> input, const F& func)
{
  if constexpr(N>0)
  {
    assert(output.sizes[0] == input.sizes[0]);
    for(size_t i=0; i<input.sizes[0]; ++i)
      copy(output[i], input[i], func);
  }
  else
    output.data[0] = func(input.data[0]);
}

template<typename T, typename U, size_t N, typename F>
void copy_add(Block<T, N> output, Block<U, N> input, const F& func)
{
  if constexpr(N>0)
  {
    assert(output.sizes[0] == input.sizes[0]);
    for(size_t i=0; i<input.sizes[0]; ++i)
      copy_add(output[i], input[i], func);
  }
  else
    output.data[0] += func(input.data[0]);
}

// What this is doing basically just copying subblock and zero filling all other
template<typename T, typename U, size_t N>
void pad(Block<T, N> output, Block<U, N> input, const T& value, Direction direction)
{
  if constexpr(N>0)
  {
    assert(output.sizes[0] >= input.sizes[0]);
    switch(direction)
    {
    case Direction::LEFT:
      for(size_t i=0; i<output.sizes[0] - input.sizes[0]; ++i)
        fill(output[i], value);

      for(size_t i=output.sizes[0] - input.sizes[0]; i<output.sizes[0]; ++i)
        pad(output[i], input[i - (output.sizes[0] - input.sizes[0])], value, direction);

      break;
    case Direction::RIGHT:
      for(size_t i=0; i<input.sizes[0]; ++i)
        pad(output[i], input[i], value, direction);

      for(size_t i=input.sizes[0]; i<output.sizes[0]; ++i)
        fill(output[i], value);

      break;
    }
  }
  else
    output.data[0] = input.data[0];
}
