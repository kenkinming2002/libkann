#include <libtensor/Norm.hpp>

namespace tensor
{
  template float       norm(Tensor<const float>       a);
  template double      norm(Tensor<const double>      a);
  template long double norm(Tensor<const long double> a);
}
