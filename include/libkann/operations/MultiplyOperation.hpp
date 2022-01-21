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
    VariableHandle gradientsImpl(VariableHandle gradient, VariableHandle input) const override;

  private:
    double m_val;
  };
}

