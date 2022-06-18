#pragma once

#include <libkann/Export.hpp>

#include <libkann/Types.hpp>
#include <libkann/Tensor.hpp>

#include <vector>

namespace kann
{
  KANN_EXPORT std::vector<Tensor> predict(Layer& layer, Executor& executor, const std::vector<Tensor>& inputs);
}
