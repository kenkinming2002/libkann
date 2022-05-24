#pragma once

#include <libkann/Tag.hpp>
#include <libkann/Task.hpp>

#include <vector>
#include <memory>

namespace kann
{
  struct Layer;
  struct LayerDef;
  struct Tensor;

  class Optimizer;
  class Executor;

  struct BatchOptimizeInput
  {
    Layer& layer;
    Tag tag;
    std::shared_ptr<const Optimizer> optimizer;
    std::vector<std::shared_ptr<const Tensor>> inputs;
    std::vector<std::shared_ptr<const Tensor>> expected_outputs;
  };

  struct BatchOptimizeOutput
  {
    std::vector<std::shared_ptr<const Tensor>> outputs;
  };

  struct OptimizeState
  {
    std::vector<std::shared_ptr<const Tensor>> values;
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

  Task<void, OptimizeInfo> optimize(Layer& layer, Tag tag, std::shared_ptr<const Optimizer> optimizer,
      const std::vector<std::shared_ptr<const Tensor>>& inputs,
      const std::vector<std::shared_ptr<const Tensor>>& expected_outputs,
      size_t batch_size, Executor& executor);
}
