#pragma once

#include <libtensor/Export.hpp>
#include <libtensor/Tensor.hpp>

namespace tensor
{
  // The raw C-like API - no template magic
  LIBTENSOR_EXPORT void split_outer_raw(size_t M, size_t N, char* __restrict__ * __restrict__ outputs, const char* __restrict__ input, size_t size) noexcept;
  LIBTENSOR_EXPORT void split_inner_raw(size_t M, size_t N, char* __restrict__ * __restrict__ outputs, const char* __restrict__ input, size_t size) noexcept;

  // The raw C++ API - do not manage any dynamic memory
  //
  // IDK, maybe I am invoking UB here, but this should work on any reasonable implementation
  template<typename T>
  inline void split_outer(size_t M, size_t N, T** output, const T* inputs) noexcept
  {
    split_outer_raw(M, N, reinterpret_cast<char**>(output), reinterpret_cast<const char*>(inputs), sizeof(T));
  }

  template<typename T>
  inline void split_inner(size_t M, size_t N, T* output, const T** inputs) noexcept
  {
    split_inner_raw(M, N, reinterpret_cast<char**>(output), reinterpret_cast<const char*>(inputs), sizeof(T));
  }

  // The API wraped in a Tensor type, hopefully, you are using these
  template<typename T>
  std::vector<Tensor<T>> split_outer(Tensor<T> value)
  {
    const auto& shape     = value.shape;
    const auto& sub_shape = shape.drop_front(1);
    const size_t M = shape.dimension(0);
    const size_t N = sub_shape.size();

    auto buffer_results = ranges::views::generate_n([&]() { return std::make_shared<Buffer<T>>(N); }, M) | ranges::to_vector;
    auto buffer_value   = value.buffer;

    // Raw API
    {
      T** outputs = new T*[buffer_results.size()];;
      const T* input;

      for(size_t i=0; i<buffer_results.size(); ++i)
        outputs[i] = buffer_results[i]->data().data();
      input = buffer_value->data().data();

      split_outer(M, N, outputs, input);

      delete[] outputs;
    }

    return buffer_results
      | ranges::views::transform([&](auto&& buffer) { return Tensor<T>(Shape::make(sub_shape), std::move(buffer)); })
      | ranges::to_vector;
  }

  template std::vector<Tensor<float>> split_outer(Tensor<float> value);

  template<typename T>
  std::vector<Tensor<T>> split_inner(Tensor<T> value)
  {
    const auto& shape     = value.shape;
    const auto& sub_shape = shape.drop_back(1);
    const size_t M = shape.dimension(shape.rank() - 1);
    const size_t N = sub_shape.size();

    auto buffer_results = ranges::views::generate_n([&]() { return std::make_shared<Buffer<T>>(N); }, M) | ranges::to_vector;
    auto buffer_value   = value.buffer;

    // Raw API
    {
      T** outputs = new T*[buffer_results.size()];;
      const T* input;

      for(size_t i=0; i<buffer_results.size(); ++i)
        outputs[i] = buffer_results[i]->data().data();
      input = buffer_value->data().data();

      split_outer(M, N, outputs, input);

      delete[] outputs;
    }

    return buffer_results
      | ranges::views::transform([&](auto&& buffer) { return Tensor<T>(Shape::make(sub_shape), std::move(buffer)); })
      | ranges::to_vector;
  }
}


