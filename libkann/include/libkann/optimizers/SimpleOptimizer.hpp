#pragma once

#include <libkann/Export.hpp>

#include <libkann/Optimizer.hpp>

namespace kann
{
  class LIBKANN_EXPORT SimpleOptimizer : public Optimizer
  {
  public:
    LIBKANN_EXPORT SimpleOptimizer() = default;
    LIBKANN_EXPORT SimpleOptimizer(float learningRate);

  public:
    LIBKANN_EXPORT void optimize(Variable& variable) const override;

  private:
    float m_learningRate;
  };
}
