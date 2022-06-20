#pragma once

#include <libkann/Tensor.hpp>

#include <vector>

namespace kann
{
  std::vector<Tensor> batch(const std::vector<Tensor>& values, size_t batch_size);
  std::vector<Tensor> unbatch(const std::vector<Tensor>& values, size_t batch_size);
}
