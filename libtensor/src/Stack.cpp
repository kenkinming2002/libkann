#include <libtensor/Stack.hpp>

// What is a math function
// There is a problem, and that is that raw pointer is much easier to work with
// as it does not carry any uncessary information

namespace tensor
{
  // Stact input of size N to output of size M*N
  template<typename T>
  void stack_outer_raw(size_t M, size_t N, T* output, const T* inputs[])
  {
    for(size_t m=0; m<M; ++m)
      for(size_t n=0; n<N; ++n)
        output[m*N+n] = inputs[m][n];
  }

  template<typename T>
  void stack_inner_raw(size_t M, size_t N, T* output, const T* inputs[])
  {
    for(size_t m=0; m<M; ++m)
      for(size_t n=0; n<N; ++n)
        output[n*M+m] = inputs[m][n];
  }

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
    std::vector<const T*> inputs = values
      | ranges::views::transform([](const auto& value) { return value.data(); })
      | ranges::to_vector;

    stack_outer_raw(M, N, output, inputs.data());

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
    std::vector<const T*> inputs = values
      | ranges::views::transform([](const auto& value) { return value.data(); })
      | ranges::to_vector;

    stack_inner_raw(M, N, output, inputs.data());

    return result;
  }

  template Tensor<const float> stack_outer(std::vector<Tensor<const float>> values);
  template Tensor<const float> stack_inner(std::vector<Tensor<const float>> values);

  template Tensor<const double> stack_outer(std::vector<Tensor<const double>> values);
  template Tensor<const double> stack_inner(std::vector<Tensor<const double>> values);

  template Tensor<const long double> stack_outer(std::vector<Tensor<const long double>> values);
  template Tensor<const long double> stack_inner(std::vector<Tensor<const long double>> values);
}


