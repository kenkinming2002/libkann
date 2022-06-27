#pragma once

#include <libkann/Export.hpp>

#include <libkann/Tensor.hpp>

namespace kann::utils
{
  KANN_EXPORT size_t max_coeff(Tensor<const float> value);
}
