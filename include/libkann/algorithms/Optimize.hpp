#pragma once

#include <libkann/Types.hpp>
#include <libkann/Tag.hpp>

#include <vector>

namespace kann
{
  void optimize(Layer& layer, Tag tag, const Optimizer& optimizer, Executor& executor, const std::vector<tensor_t>& inputs, const std::vector<tensor_t>& expected_outputs);
}
