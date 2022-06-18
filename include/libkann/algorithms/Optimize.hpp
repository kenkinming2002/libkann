#pragma once

#include <libkann/Export.hpp>

#include <libkann/Types.hpp>
#include <libkann/Tensor.hpp>
#include <libkann/Tag.hpp>

#include <vector>

namespace kann
{
  KANN_EXPORT void optimize(Layer& layer, Tag tag, const Optimizer& optimizer, Executor& executor, size_t batch_size, const std::vector<Tensor>& inputs, const std::vector<Tensor>& expected_outputs);
}
