#pragma once

#include <libkann/Export.hpp>
#include <libkann/Tensor.hpp>

#include <vector>

namespace kann
{
  KANN_EXPORT std::vector<Tensor<float>> batch(const std::vector<Tensor<float>>& values, size_t batch_size);
  KANN_EXPORT std::vector<Tensor<float>> unbatch(const std::vector<Tensor<float>>& values, size_t batch_size);
}
