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

  struct BatchPredictInput
  {
    Layer& layer;
    std::vector<std::shared_ptr<const Tensor>> inputs;
  };

  struct BatchPredictOutput
  {
    std::vector<std::shared_ptr<const Tensor>> outputs;
  };

  BatchPredictOutput batch_predict(const BatchPredictInput& input, Executor& executor);

  struct PredictInfo
  {
    const Tensor& input;
    const Tensor& output;
  };

  Task<void, PredictInfo> predict(Layer& layer,
      const std::vector<std::shared_ptr<const Tensor>>& inputs,
      size_t batch_size, Executor& executor);
}
