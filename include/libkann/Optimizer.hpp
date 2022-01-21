#pragma once

#include <libkann/Model.hpp>
#include <libkann/Executor.hpp>

namespace kann
{
  class Optimizer
  {
  public:
    Optimizer() = default;
    Optimizer(std::shared_ptr<Model> model, double learningRate, size_t batchSize = 1);

  public:
    /* @return [output, cost] */
    std::vector<std::pair<std::shared_ptr<const Tensor>, double>> optimize(
      std::vector<std::shared_ptr<const Tensor>> inputs,
      std::vector<std::shared_ptr<const Tensor>> expectedOutputs,
      unsigned tag = TAG_ALL);

  private:
    std::shared_ptr<Model> m_model;
    double m_learningRate;
    size_t m_batchSize;

    std::unique_ptr<Executor> m_executor;
    std::vector<std::shared_ptr<const Tensor>> m_statesValues;
  };
}

