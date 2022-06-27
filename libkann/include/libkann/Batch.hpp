#pragma once

#include <libkann/Export.hpp>
#include <libtensor/Tensor.hpp>

#include <vector>

namespace kann
{
  KANN_EXPORT std::vector<tensor::Tensor<const float>> batch(std::vector<tensor::Tensor<const float>> values, size_t batch_size);
  KANN_EXPORT std::vector<tensor::Tensor<const float>> unbatch(std::vector<tensor::Tensor<const float>> values, size_t batch_size);
}
