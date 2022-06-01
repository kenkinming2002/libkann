#pragma once

#include <libkann/Operation.hpp>

namespace kann
{
  class ScaleOperation : public OperationImpl<ScaleOperation, 1>
  {
  public:
    ScaleOperation(double val);

  public:
    Tensor process_impl(inputs_t inputs) const;
    variables_t gradients_impl(variable_t gradient, variables_t inputs) const;

  private:
    double m_val;
  };
}

