#pragma once

#include <libkann/Model.hpp>
#include <libkann/Executor.hpp>

namespace kann
{
  class Predictor
  {
  public:
    Predictor() = default;
    Predictor(std::shared_ptr<Model> model);

  public:
    std::shared_ptr<const Tensor> predict(std::shared_ptr<const Tensor> input);

  private:
    std::shared_ptr<const Model> m_model;
    Executor m_executor;

    std::vector<std::shared_ptr<const Tensor>> m_state;
  };
}

