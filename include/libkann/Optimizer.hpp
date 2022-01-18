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
    std::pair<Tensor, double> optimize(Tensor input, Tensor expectedOutput);

  private:
    std::shared_ptr<Model> m_model;
    double m_learningRate;

    Executor m_executor;

    std::vector<Tensor> m_state;
  };
}

