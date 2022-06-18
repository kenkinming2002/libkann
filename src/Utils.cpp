#include <libkann/Utils.hpp>

#include <Eigen/Eigen>

namespace kann::utils
{
  using EigenArray  = Eigen::ArrayXd;
  using EigenMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  template<typename Storage>
  static inline auto to_eigen_array(const TensorBase<Storage>& tensor)
  {
    return EigenArray::Map(tensor.data(), tensor.size());
  }

  template<typename Storage>
  static inline auto to_eigen_matrix(const TensorBase<Storage>& tensor)
  {
    assert(tensor.is_matrix());
    return EigenMatrix::Map(tensor.data(),
      tensor.shape().dimension(0),
      tensor.shape().dimension(1)
    );
  }

  size_t max_coeff(TensorRef value)
  {
    size_t coeff;
    to_eigen_array(value).maxCoeff(&coeff);
    return coeff;
  }

  void copy(TensorRef from, MutableTensorRef to)
  {
    assert(from.shape() == to.shape());
    ranges::copy_n(from.data(), from.size(), to.data());
  }
}
