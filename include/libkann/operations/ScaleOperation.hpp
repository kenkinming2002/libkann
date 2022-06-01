#pragma once

#include <libkann/Operation.hpp>

namespace kann
{
  class ScaleOperation : public UnaryOperation
  {
  public:
    ScaleOperation(double val);

  public:
    Tensor processImpl(const Tensor& input) const override;
    variable_t gradientsImpl(variable_t gradient, variable_t input) const override;

  private:
    double m_val;
  };
}

