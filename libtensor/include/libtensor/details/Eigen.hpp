#pragma once

#include <libtensor/Tensor.hpp>

#include <Eigen/Eigen>
#include <unsupported/Eigen/CXX11/Tensor>

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

  template<typename T>
  static inline auto to_tensor4d(Tensor<T> value, bool trans)
  {
    using TensorType = std::conditional_t<std::is_const_v<T>,
      Eigen::Tensor<std::remove_const_t<T>, 4, Eigen::RowMajor> const,
      Eigen::Tensor<std::remove_const_t<T>, 4, Eigen::RowMajor>>;

    typename TensorType::Index dim0 = value.dimension(0);
    typename TensorType::Index dim1 = value.dimension(1);
    typename TensorType::Index dim2 = value.dimension(2);
    typename TensorType::Index dim3 = value.dimension(3);

    assert(value.rank() == 4);
    if (trans) return Eigen::TensorMap<TensorType>(value.data(), dim0, dim1, dim2, dim3).shuffle(Eigen::array<int, 4>{1,0,2,3});
    else       return Eigen::TensorMap<TensorType>(value.data(), dim0, dim1, dim2, dim3).shuffle(Eigen::array<int, 4>{0,1,2,3});
  }
}
