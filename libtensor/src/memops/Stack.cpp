#include <libtensor/memops/Stack.hpp>

namespace tensor
{
  template<typename T>
  static inline void stack_verify_same_shape(const std::vector<Tensor<T>>& values)
  {
    if(values.empty())
      throw std::runtime_error("Cannot stack empty vector of tensors");

    for(const auto& value : values)
      if(values.front().shape != value.shape)
        throw std::runtime_error("All shapes must be the same for stacking");
  }

  template<typename T>
  Tensor<T> stack_outer(std::vector<Tensor<T>> values)
  {
    stack_verify_same_shape(values);

    const auto& shape = values.front().shape;
    const size_t M = values.size();
    const size_t N = shape.size();

    auto result_buffer = std::make_shared<Buffer<T>>(M * N);
    for(size_t m=0; m<M; ++m)
      for(size_t n=0; n<N; ++n)
      {
        const auto& value_buffer = values[m].buffer;
        (*result_buffer)[m*N+n] = (*value_buffer)[n];
      }

    return Tensor<T>(Shape::make(M, shape), result_buffer);
  }

  template<typename T>
  Tensor<T> stack_inner(std::vector<Tensor<T>> values)
  {
    stack_verify_same_shape(values);

    const auto& shape = values.front().shape;
    const size_t M = shape.size();
    const size_t N = values.size();

    auto result_buffer = std::make_shared<Buffer<T>>(M * N);
    for(size_t m=0; m<M; ++m)
      for(size_t n=0; n<N; ++n)
      {
        const auto& value_buffer = values[n].buffer;
        (*result_buffer)[m*N+n] = (*value_buffer)[m];
      }

    return Tensor<T>(Shape::make(M, shape), result_buffer);
  }

  template Tensor<float> stack_outer(std::vector<Tensor<float>> values);
  template Tensor<double> stack_outer(std::vector<Tensor<double>> values);
  template Tensor<long double> stack_outer(std::vector<Tensor<long double>> values);

  template Tensor<float> stack_inner(std::vector<Tensor<float>> values);
  template Tensor<double> stack_inner(std::vector<Tensor<double>> values);
  template Tensor<long double> stack_inner(std::vector<Tensor<long double>> values);
}


