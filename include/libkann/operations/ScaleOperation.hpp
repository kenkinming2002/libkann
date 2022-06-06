#pragma once

#include <libkann/operations/CWiseOperation.hpp>

namespace kann
{
  class ScaleOperation : public CWiseOperation<ScaleOperation, 1, 1>
  {
  public:
    ScaleOperation(size_t size, double val);

  public:
    cwise_outputs_t forward(cwise_inputs_t inputs) const;
    cwise_inputs_t backward(cwise_inputs_t inputs, cwise_outputs_t output_gradients) const;

  private:
    double m_val;
  };
}

