#include <libtensor/Reduce.hpp>

#include <libtensor/details/Eigen.hpp>

namespace tensor
{
  template<Direction direction, typename T>
  Tensor<T> reduce(Tensor<const T> value)
  {
    auto [M, N] = std::make_pair(value.dimension(0), value.dimension(1));
    if constexpr(direction == Direction::LEFT)
    {
      auto result = Tensor<T>::create(Shape::make(N));
      details::to_array1d(result) = details::to_array2d(value).colwise().sum();
      return result;
    }
    else if constexpr(direction == Direction::RIGHT)
    {
      auto result = Tensor<T>::create(Shape::make(M));
      details::to_array1d(result) = details::to_array2d(value).rowwise().sum();
      return result;
    }
    else
      static_assert("Invalid direction");
  }

  template Tensor<float> reduce<Direction::LEFT> (Tensor<const float> value);
  template Tensor<float> reduce<Direction::RIGHT>(Tensor<const float> value);

  template Tensor<double> reduce<Direction::LEFT> (Tensor<const double> value);
  template Tensor<double> reduce<Direction::RIGHT>(Tensor<const double> value);

  template Tensor<long double> reduce<Direction::LEFT> (Tensor<const long double> value);
  template Tensor<long double> reduce<Direction::RIGHT>(Tensor<const long double> value);
}
