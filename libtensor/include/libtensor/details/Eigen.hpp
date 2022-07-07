#pragma once

#include <libtensor/Tensor.hpp>
#include <Eigen/Eigen>

namespace tensor::details
{
  template<typename T>
  static inline auto to_array1d(Tensor<T> value)
  {
    using ArrayType = std::conditional_t<std::is_const_v<T>,
      Eigen::Array<std::remove_const_t<T>, 1, Eigen::Dynamic> const,
      Eigen::Array<std::remove_const_t<T>, 1, Eigen::Dynamic>>;

    assert(value.rank() == 1);
    return ArrayType::Map(value.data(), value.dimension(0));
  }

  template<typename T>
  static inline auto to_array2d(Tensor<T> value)
  {
    using ArrayType = std::conditional_t<std::is_const_v<T>,
      Eigen::Array<std::remove_const_t<T>, Eigen::Dynamic, Eigen::Dynamic> const,
      Eigen::Array<std::remove_const_t<T>, Eigen::Dynamic, Eigen::Dynamic>>;

    assert(value.rank() == 2);
    return ArrayType::Map(value.data(), value.dimension(0), value.dimension(1));
  }

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
}
