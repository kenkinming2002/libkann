#pragma once

#include <libkann/Types.hpp>
#include <libkann/Tensor.hpp>

#include <vector>

namespace kann
{
  std::vector<Tensor> predict(Layer& layer, Executor& executor, const std::vector<Tensor>& inputs);
}
