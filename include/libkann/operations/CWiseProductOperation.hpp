#pragma once

#include <libkann/Operation.hpp>

namespace kann
{
  class CWiseProductOperation : public BinaryOperation
  {
  public:
    Tensor processImpl(const Tensor& a, const Tensor& b) const override;
    std::pair<CRef<Variable>, CRef<Variable>> gradientsImpl(CRef<Variable> gradient, CRef<Variable> input1, CRef<Variable> input2) const override;
  };
}

