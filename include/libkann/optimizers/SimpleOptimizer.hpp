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
    ProcessOutput process(ProcessInput input) const override;

  private:
    double m_learningRate;
  };
}
