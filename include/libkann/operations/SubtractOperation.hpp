#pragma once

#include <libkann/Operation.hpp>

namespace kann
{
  class SubtractOperation : public OperationImpl<SubtractOperation, 2>
  {
  public:
    Tensor process_impl(inputs_t inputs) const;
    variables_t gradients_impl(variable_t gradient, variables_t inputs) const;
  };
}


