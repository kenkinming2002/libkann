#pragma once

#include <libkann/Tensor.hpp>
#include <libkann/NewLayer.hpp>
#include <libkann/Executor.hpp>

#include <vector>
#include <memory>

namespace kann
{
  class NewModel
  {
  public:
    NewModel(std::shared_ptr<const NewLayer> layer);

  public:
    std::shared_ptr<const Tensor> predict(std::shared_ptr<const Tensor> input);

  private:
    std::shared_ptr<const NewLayer> m_layer;

    std::vector<std::shared_ptr<const Tensor>> m_parameters;
    std::vector<std::shared_ptr<const Tensor>> m_states;

    std::unique_ptr<Executor> m_predictExecutor;
  };
}
