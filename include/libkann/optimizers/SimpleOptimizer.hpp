#pragma once

#include <libkann/Export.hpp>

#include <libkann/Optimizer.hpp>

namespace kann
{
  class SimpleOptimizer : public Optimizer
  {
  public:
    KANN_EXPORT SimpleOptimizer() = default;
    KANN_EXPORT SimpleOptimizer(float learningRate);

  public:
    KANN_EXPORT size_t process(Graph& graph, Info& info, Shape shape, size_t index, size_t gradient_index) const override;

  private:
    float m_learningRate;
  };
}
