#pragma once

#include <libkann/Export.hpp>
#include <libtensor/Tensor.hpp>

#include <vector>

namespace kann
{
  KANN_EXPORT std::vector<Tensor<const float>> batch(std::vector<Tensor<const float>> values, size_t batch_size);
  KANN_EXPORT std::vector<Tensor<const float>> unbatch(std::vector<Tensor<const float>> values, size_t batch_size);
}
