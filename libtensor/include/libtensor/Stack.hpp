#pragma once

#include <libtensor/Export.hpp>
#include <libtensor/Tensor.hpp>

namespace tensor
{
  // The raw C-like API - no template magic
  LIBTENSOR_EXPORT void stack_outer_raw(size_t M, size_t N, char* __restrict__ output, const char* __restrict__ * __restrict__ inputs, size_t size) noexcept;
  LIBTENSOR_EXPORT void stack_inner_raw(size_t M, size_t N, char* __restrict__ output, const char* __restrict__ * __restrict__ inputs, size_t size) noexcept;

  // The raw C++ API - do not manage any dynamic memory
  //
  // IDK, maybe I am invoking UB here, but this should work on any reasonable implementation
  template<typename T>
  inline void stack_outer(size_t M, size_t N, T* output, const T** inputs) noexcept
  {
    stack_outer_raw(M, N, reinterpret_cast<char*>(output), reinterpret_cast<const char**>(inputs), sizeof(T));
  }

  template<typename T>
  inline void stack_inner(size_t M, size_t N, T* output, const T** inputs) noexcept
  {
    stack_inner_raw(M, N, reinterpret_cast<char*>(output), reinterpret_cast<const char**>(inputs), sizeof(T));
  }

  // The API wraped in a Tensor type, hopefully, you are using these
  template<typename T>
  Tensor<const T> stack_outer(std::vector<Tensor<const T>> values)
  {
    if(values.empty() || std::any_of(values.begin(), values.end(), [&](const auto& value) { return values.front().shape() != value.shape(); }))
      throw std::runtime_error("All shapes must be the same for stacking");

    const auto& shape = values.front().shape();
    const size_t M = values.size();
    const size_t N = shape.size();
    auto result = Tensor<T>::create(Shape::make(M, shape));

    T* output = result.data();
    const T** inputs = new const T*[values.size()];

    for(size_t i=0; i<values.size(); ++i)
      inputs[i] = values[i].data();

    stack_outer(M, N, output, inputs);

    delete[] inputs;

    return result;
  }

  template<typename T>
  Tensor<const T> stack_inner(std::vector<Tensor<const T>> values)
  {
    if(values.empty() || std::any_of(values.begin(), values.end(), [&](const auto& value) { return values.front().shape() != value.shape(); }))
      throw std::runtime_error("All shapes must be the same for stacking");

    const auto& shape = values.front().shape();
    const size_t M = values.size();
    const size_t N = shape.size();

    auto result = Tensor<T>::create(Shape::make(M, shape));

    T* output = result.data();
    const T** inputs = new const T*[values.size()];

    for(size_t i=0; i<values.size(); ++i)
      inputs[i] = values[i].data();

    stack_inner(M, N, output, inputs);

    delete[] inputs;

    return result;
  }
}

