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
    Tensor predict(Tensor input);

  private:
    std::shared_ptr<const Model> m_model;
    Executor m_executor;

    std::vector<Tensor> m_state;
  };
}

