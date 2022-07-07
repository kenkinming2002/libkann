#include <libtensor/Broadcast.hpp>

namespace tensor
{
  template Tensor<float> broadcast_add<Direction::LEFT>(Tensor<const float> a, Tensor<const float> b);
  template Tensor<float> broadcast_sub<Direction::LEFT>(Tensor<const float> a, Tensor<const float> b);
  template Tensor<float> broadcast_mul<Direction::LEFT>(Tensor<const float> a, Tensor<const float> b);
  template Tensor<float> broadcast_div<Direction::LEFT>(Tensor<const float> a, Tensor<const float> b);

  template Tensor<float> broadcast_add<Direction::RIGHT>(Tensor<const float> a, Tensor<const float> b);
  template Tensor<float> broadcast_sub<Direction::RIGHT>(Tensor<const float> a, Tensor<const float> b);
  template Tensor<float> broadcast_mul<Direction::RIGHT>(Tensor<const float> a, Tensor<const float> b);
  template Tensor<float> broadcast_div<Direction::RIGHT>(Tensor<const float> a, Tensor<const float> b);

  template Tensor<double> broadcast_add<Direction::LEFT>(Tensor<const double> a, Tensor<const double> b);
  template Tensor<double> broadcast_sub<Direction::LEFT>(Tensor<const double> a, Tensor<const double> b);
  template Tensor<double> broadcast_mul<Direction::LEFT>(Tensor<const double> a, Tensor<const double> b);
  template Tensor<double> broadcast_div<Direction::LEFT>(Tensor<const double> a, Tensor<const double> b);

  template Tensor<double> broadcast_add<Direction::RIGHT>(Tensor<const double> a, Tensor<const double> b);
  template Tensor<double> broadcast_sub<Direction::RIGHT>(Tensor<const double> a, Tensor<const double> b);
  template Tensor<double> broadcast_mul<Direction::RIGHT>(Tensor<const double> a, Tensor<const double> b);
  template Tensor<double> broadcast_div<Direction::RIGHT>(Tensor<const double> a, Tensor<const double> b);

  template Tensor<long double> broadcast_add<Direction::LEFT>(Tensor<const long double> a, Tensor<const long double> b);
  template Tensor<long double> broadcast_sub<Direction::LEFT>(Tensor<const long double> a, Tensor<const long double> b);
  template Tensor<long double> broadcast_mul<Direction::LEFT>(Tensor<const long double> a, Tensor<const long double> b);
  template Tensor<long double> broadcast_div<Direction::LEFT>(Tensor<const long double> a, Tensor<const long double> b);

  template Tensor<long double> broadcast_add<Direction::RIGHT>(Tensor<const long double> a, Tensor<const long double> b);
  template Tensor<long double> broadcast_sub<Direction::RIGHT>(Tensor<const long double> a, Tensor<const long double> b);
  template Tensor<long double> broadcast_mul<Direction::RIGHT>(Tensor<const long double> a, Tensor<const long double> b);
  template Tensor<long double> broadcast_div<Direction::RIGHT>(Tensor<const long double> a, Tensor<const long double> b);
}
