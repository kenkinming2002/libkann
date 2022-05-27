#pragma once

#include <libkann/Types.hpp>
#include <libkann/Tag.hpp>
#include <libkann/Task.hpp>

#include <vector>

namespace kann
{
  struct BatchOptimizeInput
  {
    Layer& layer;
    Tag tag;
    optimizer_t optimizer;
    std::vector<tensor_t> inputs;
    std::vector<tensor_t> expected_outputs;
  };

  struct BatchOptimizeOutput
  {
    std::vector<tensor_t> outputs;
  };

  struct OptimizeState
  {
    std::vector<tensor_t> values;
  };

  OptimizeState create_optimize_state(const LayerDef& def, Tag tag, const Optimizer& optimizer);
  BatchOptimizeOutput batch_optimize(const BatchOptimizeInput& input, OptimizeState& state, Executor& executor);

  struct OptimizeInfo
  {
    const Tensor& input;
    const Tensor& output;
    const Tensor& expected_output;
    double cost;
  };

  Task<void, OptimizeInfo> optimize(Layer& layer, Tag tag, optimizer_t optimizer,
      const std::vector<tensor_t>& inputs,
      const std::vector<tensor_t>& expected_outputs,
      size_t batch_size, Executor& executor);
}
