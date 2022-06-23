#include <libkann/Utils.hpp>

#include <Eigen/Eigen>

namespace kann::utils
{
  using EigenArray  = Eigen::ArrayXf;
  using EigenMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  template<typename T>
  static inline auto to_eigen_array(const TensorRef<T>& tensor)
  {
    return EigenArray::Map(tensor.data(), tensor.size());
  }

  size_t max_coeff(TensorRef<const float> value)
  {
    size_t coeff;
    to_eigen_array(value).maxCoeff(&coeff);
    return coeff;
  }
}
