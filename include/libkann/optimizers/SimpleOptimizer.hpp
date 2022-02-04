#pragma once

#include <libkann/Optimizer.hpp>

namespace kann
{
  class SimpleOptimizer : public Optimizer
  {
  public:
    SimpleOptimizer(double learningRate);

  public:
    Result process(VRef parameter, VRef gradient, VMap state) const override;

  private:
    double m_learningRate;
  };
}
