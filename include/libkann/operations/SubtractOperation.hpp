#pragma once

#include <libkann/Operation.hpp>

namespace kann
{
  class SubtractOperation : public BinaryOperation
  {
  public:
    Tensor processImpl(const Tensor& a, const Tensor& b) const override;
    VariablePair gradientsImpl(VariableHandle gradient, VariableHandle a, VariableHandle b) const override;
  };
}


