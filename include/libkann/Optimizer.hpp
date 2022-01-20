#pragma once

#include <libkann/Model.hpp>
#include <libkann/Executor.hpp>

namespace kann
{
  class Optimizer
  {
  public:
    Optimizer() = default;
    Optimizer(std::shared_ptr<Model> model, double learningRate);

  public:
    /* @return [output, cost] */
    std::pair<std::shared_ptr<const Tensor>, double> optimize(std::shared_ptr<const Tensor> input, std::shared_ptr<const Tensor> expectedOutput, unsigned tag = TAG_ALL);

  private:
    std::shared_ptr<Model> m_model;
    double m_learningRate;

    std::unique_ptr<Executor> m_executor;
    std::vector<std::shared_ptr<const Tensor>> m_state;
  };
}

