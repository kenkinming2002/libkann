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
    KANN_EXPORT void optimize(Variable& variable) const override;

  private:
    float m_learningRate;
  };
}
