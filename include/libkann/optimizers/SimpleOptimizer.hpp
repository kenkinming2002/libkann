#pragma once

#include <libkann/Optimizer.hpp>

namespace kann
{
  class SimpleOptimizer : public Optimizer
  {
  public:
    SimpleOptimizer() = default;
    SimpleOptimizer(double learningRate);

  public:
    size_t process(Graph& graph, Info& info, Shape shape, size_t index, size_t gradient_index) const override;

  private:
    double m_learningRate;
  };
}
