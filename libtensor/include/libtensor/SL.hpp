#pragma once

#include <libtensor/Export.hpp>
#include <libtensor/Tensor.hpp>

#include <string>

namespace tensor
{
  LIBTENSOR_EXPORT void save_tensor(tensor::Tensor<float> value, const std::string& filename);
  LIBTENSOR_EXPORT tensor::Tensor<float> load_tensor(const std::string& filename);
}
