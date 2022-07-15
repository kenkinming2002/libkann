#include <libtensor/Reduce.hpp>

#include <libtensor/details/Eigen.hpp>

namespace tensor
{
  template<typename T>
  Tensor<T> reduce_outer(Tensor<const T> value)
  {
    auto [M, N] = std::make_pair(value.dimension(0), value.dimension(1));
    auto result = Tensor<T>::create(Shape::make(N));
    details::to_array1d(result) = details::to_array2d(value).colwise().sum();
    return result;
  }

  template<typename T>
  Tensor<T> reduce_inner(Tensor<const T> value)
  {
    auto [M, N] = std::make_pair(value.dimension(0), value.dimension(1));
    auto result = Tensor<T>::create(Shape::make(M));
    details::to_array1d(result) = details::to_array2d(value).rowwise().sum();
    return result;
  }

  template Tensor<float> reduce_outer(Tensor<const float> value);
  template Tensor<float> reduce_inner(Tensor<const float> value);

  template Tensor<double> reduce_outer(Tensor<const double> value);
  template Tensor<double> reduce_inner(Tensor<const double> value);

  template Tensor<long double> reduce_outer(Tensor<const long double> value);
  template Tensor<long double> reduce_inner(Tensor<const long double> value);
}
