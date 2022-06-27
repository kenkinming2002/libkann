#include <libtensor/Utils.hpp>

#include <Eigen/Eigen>

namespace tensor::utils
{
  size_t max_coeff(Tensor<const float> value)
  {
    size_t coeff;
    Eigen::ArrayXf::Map(value.data(), value.size()).maxCoeff(&coeff);
    return coeff;
  }
}
