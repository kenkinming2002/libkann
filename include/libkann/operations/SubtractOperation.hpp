#pragma once

#include <libkann/operations/CWiseOperation.hpp>

namespace kann
{
  class SubtractOperation : public CWiseOperation<SubtractOperation, 2, 1>
  {
  public:
    SubtractOperation(size_t size);

  public:
    cwise_outputs_t forward(cwise_inputs_t inputs) const;
    cwise_inputs_t backward(cwise_inputs_t inputs, cwise_outputs_t output_gradients) const;
  };
}


