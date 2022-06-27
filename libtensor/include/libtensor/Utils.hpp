#pragma once

#include <libtensor/Export.hpp>

#include <libtensor/Tensor.hpp>

namespace tensor::utils
{
  LIBTENSOR_EXPORT size_t max_coeff(Tensor<const float> value);
}
