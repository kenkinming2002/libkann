#pragma once

// Declaration
#include <libtensor/Export.hpp>
#include <libtensor/Tensor.hpp>

namespace tensor
{
  template<typename T>
  Tensor<T> matrix_product(Tensor<const T> a, bool trans_a, Tensor<const T> b, bool trans_b);

  extern template LIBTENSOR_EXPORT Tensor<float>       matrix_product(Tensor<const float>       a, bool trans_a, Tensor<const float>  b,      bool trans_b);
  extern template LIBTENSOR_EXPORT Tensor<double>      matrix_product(Tensor<const double>      a, bool trans_a, Tensor<const double> b,      bool trans_b);
  extern template LIBTENSOR_EXPORT Tensor<long double> matrix_product(Tensor<const long double> a, bool trans_a, Tensor<const long double> b, bool trans_b);
}

// Implementation
#include <Eigen/Eigen>

namespace tensor
{
  template<bool Trans, typename T>
  static inline auto to_matrix(Tensor<T> value)
  {
    using MatrixType = std::conditional_t<std::is_const_v<T>,
      Eigen::Matrix<std::remove_const_t<T>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> const,
      Eigen::Matrix<std::remove_const_t<T>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

    assert(value.rank() == 2);
    if constexpr (Trans) return MatrixType::Map(value.data(), value.dimension(0), value.dimension(1)).transpose();
    else                 return MatrixType::Map(value.data(), value.dimension(0), value.dimension(1));
  }

  template<typename T>
  Tensor<T> matrix_product(Tensor<const T> a, bool trans_a, Tensor<const T> b, bool trans_b)
  {
    auto [M, K1] = std::make_pair(a.dimension(0), a.dimension(1));
    auto [K2, N] = std::make_pair(a.dimension(0), a.dimension(1));

    if(trans_a) std::swap(M, K1);
    if(trans_b) std::swap(K2, N);

    auto c = Tensor<T>::create(Shape(M, N));
    if(trans_a)
    {
      if(trans_b) to_matrix<false>(c).noalias() = to_matrix<true>(a) * to_matrix<true> (b);
      else        to_matrix<false>(c).noalias() = to_matrix<true>(a) * to_matrix<false>(b);
    }
    else
    {
      if(trans_b) to_matrix<false>(c).noalias() = to_matrix<false>(a) * to_matrix<true> (b);
      else        to_matrix<false>(c).noalias() = to_matrix<false>(a) * to_matrix<false>(b);
    }
    return c;
  }
}
