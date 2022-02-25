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
    CRef<Variable> gradientsImpl(CRef<Variable> gradient, CRef<Variable> input) const override;

  private:
    double m_val;
  };
}

