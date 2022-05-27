#pragma once

#include <libkann/Types.hpp>
#include <libkann/Task.hpp>

#include <vector>

namespace kann
{
  struct BatchPredictInput
  {
    Layer& layer;
    std::vector<tensor_t> inputs;
  };

  struct BatchPredictOutput
  {
    std::vector<tensor_t> outputs;
  };

  BatchPredictOutput batch_predict(const BatchPredictInput& input, Executor& executor);

  struct PredictInfo
  {
    const Tensor& input;
    const Tensor& output;
  };

  Task<void, PredictInfo> predict(Layer& layer,
      const std::vector<tensor_t>& inputs,
      size_t batch_size, Executor& executor);
}
