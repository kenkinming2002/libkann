#pragma once

#include <array>

#include <stddef.h>

template<typename T, size_t N>
constexpr std::array<T, N-1> remove_nth(std::array<T, N> data, size_t n)
{
  std::array<T, N-1> result;
  for(size_t i=0; i<n; ++i)
    result[i] = data[i];

  for(size_t i=n+1; i<N; ++i)
    result[i-1] = data[i];

  return result;
}

template<typename T, size_t N>
constexpr std::array<T, N> replace_nth(std::array<T, N> data, size_t n, T value)
{
  data[n] = value;
  return data;
}

template<typename T, size_t N>
constexpr std::array<T, N-1> tail(std::array<T, N> data)
{
  return remove_nth(data, 0);
}

template<typename T, size_t N>
constexpr std::array<T, N> replace_head(std::array<T, N> data, T value)
{
  return replace_nth(data, 0, value);
}
