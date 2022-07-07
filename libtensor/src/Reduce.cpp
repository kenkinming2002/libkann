#include <libtensor/Reduce.hpp>

namespace tensor
{
  template Tensor<float> reduce<Direction::LEFT> (Tensor<const float> value);
  template Tensor<float> reduce<Direction::RIGHT>(Tensor<const float> value);

  template Tensor<double> reduce<Direction::LEFT> (Tensor<const double> value);
  template Tensor<double> reduce<Direction::RIGHT>(Tensor<const double> value);

  template Tensor<long double> reduce<Direction::LEFT> (Tensor<const long double> value);
  template Tensor<long double> reduce<Direction::RIGHT>(Tensor<const long double> value);
}
