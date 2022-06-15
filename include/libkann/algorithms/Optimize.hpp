#pragma once

#include <libkann/Types.hpp>
#include <libkann/Tensor.hpp>
#include <libkann/Tag.hpp>

#include <vector>

namespace kann
{
  void optimize(Layer& layer, Tag tag, const Optimizer& optimizer, Executor& executor, const std::vector<Tensor>& inputs, const std::vector<Tensor>& expected_outputs);
}
