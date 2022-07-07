#include <libtensor/MatrixProduct.hpp>

namespace tensor
{
  template Tensor<float>  matrix_product(Tensor<const float>  a, bool trans_a, Tensor<const float>  b, bool trans_b);
  template Tensor<double> matrix_product(Tensor<const double> a, bool trans_a, Tensor<const double> b, bool trans_b);
  template Tensor<long double> matrix_product(Tensor<const long double> a, bool trans_a, Tensor<const long double> b, bool trans_b);
}

