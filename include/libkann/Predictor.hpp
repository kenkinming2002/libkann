#pragma once

#include <libkann/Model.hpp>
#include <libkann/Executor.hpp>

namespace kann
{
  class Predictor
  {
  public:
    Predictor() = default;
    Predictor(std::shared_ptr<const Model> model);

  public:
    std::shared_ptr<const Tensor> predict(std::shared_ptr<const Tensor> input);

  private:
    std::shared_ptr<const Model> m_model;

    std::unique_ptr<Executor> m_executor;
    std::vector<std::shared_ptr<const Tensor>> m_statesValues;
  };
}

