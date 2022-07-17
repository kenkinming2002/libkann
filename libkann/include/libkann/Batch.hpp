#pragma once

#include <libkann/Export.hpp>
#include <libtensor/Tensor.hpp>

#include <vector>

namespace kann
{
  KANN_EXPORT std::vector<tensor::Tensor<float>> batch(std::vector<tensor::Tensor<float>> values, size_t batch_size);
  KANN_EXPORT std::vector<tensor::Tensor<float>> unbatch(std::vector<tensor::Tensor<float>> values, size_t batch_size);
}
