#pragma once

#include <libkann/Operation.hpp>

namespace kann
{
  class CWiseProductOperation : public BinaryOperation
  {
  public:
    Tensor processImpl(const Tensor& a, const Tensor& b) const override;
    VariablePair gradientsImpl(VariableHandle gradient, VariableHandle input1, VariableHandle input2) const override;
  };
}

