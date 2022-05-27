#pragma once

#include <libkann/Operation.hpp>

namespace kann
{
  class SubtractOperation : public BinaryOperation
  {
  public:
    Tensor processImpl(const Tensor& a, const Tensor& b) const override;
    std::pair<variable_t, variable_t> gradientsImpl(variable_t gradient, variable_t a, variable_t b) const override;
  };
}


