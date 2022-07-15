#include <libtensor/MatrixProduct.hpp>

#include <libtensor/details/Eigen.hpp>

namespace tensor
{
  template<typename T>
  Tensor<T> matrix_product(Tensor<const T> a, bool trans_a, Tensor<const T> b, bool trans_b)
  {
    auto [M, K1] = std::make_pair(a.dimension(0), a.dimension(1));
    auto [K2, N] = std::make_pair(b.dimension(0), b.dimension(1));

    if(trans_a) std::swap(M, K1);
    if(trans_b) std::swap(K2, N);

    auto c = Tensor<T>::create(Shape::make(M, N));
    if(trans_a)
    {
      if(trans_b) details::to_matrix<false>(c).noalias() = details::to_matrix<true>(a)  * details::to_matrix<true> (b);
      else        details::to_matrix<false>(c).noalias() = details::to_matrix<true>(a)  * details::to_matrix<false>(b);
    }
    else
    {
      if(trans_b) details::to_matrix<false>(c).noalias() = details::to_matrix<false>(a) * details::to_matrix<true> (b);
      else        details::to_matrix<false>(c).noalias() = details::to_matrix<false>(a) * details::to_matrix<false>(b);
    }
    return c;
  }

  template Tensor<float>  matrix_product(Tensor<const float>  a, bool trans_a, Tensor<const float>  b, bool trans_b);
  template Tensor<double> matrix_product(Tensor<const double> a, bool trans_a, Tensor<const double> b, bool trans_b);
  template Tensor<long double> matrix_product(Tensor<const long double> a, bool trans_a, Tensor<const long double> b, bool trans_b);
}

