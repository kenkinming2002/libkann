#pragma once

#include <libkann/Export.hpp>
#include <libkann/Tensor.hpp>

#include <vector>

namespace kann
{
  KANN_EXPORT std::vector<Tensor> batch(const std::vector<Tensor>& values, size_t batch_size);
  KANN_EXPORT std::vector<Tensor> unbatch(const std::vector<Tensor>& values, size_t batch_size);
}
