#pragma once

#include <libkann/Operation.hpp>

namespace kann
{
  class MultiplyOperation : public UnaryOperation
  {
  public:
    MultiplyOperation(double val);

  public:
    Tensor processImpl(const Tensor& input) const override;
    variable_t gradientsImpl(variable_t gradient, variable_t input) const override;

  private:
    double m_val;
  };
}

