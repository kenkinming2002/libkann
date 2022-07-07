#include <libtensor/MaxCoeff.hpp>

namespace tensor
{
  template size_t max_coeff(Tensor<const float>       a);
  template size_t max_coeff(Tensor<const double>      a);
  template size_t max_coeff(Tensor<const long double> a);
}

