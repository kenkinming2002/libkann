#pragma once

#include <libkann/Operation.hpp>

namespace kann
{
  class CWiseProductOperation : public BinaryOperation
  {
  public:
    Tensor processImpl(const Tensor& a, const Tensor& b) const override;
    std::pair<variable_t, variable_t> gradientsImpl(variable_t gradient, variable_t input1, variable_t input2) const override;
  };
}

