#pragma once

#include <libkann/Types.hpp>

#include <vector>

namespace kann
{
  std::vector<tensor_t> predict(Layer& layer, Executor& executor, const std::vector<tensor_t>& inputs);
}
