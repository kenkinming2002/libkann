#include <libkann/operations/SubtractOperation.hpp>

namespace kann
{
  SubtractOperation::SubtractOperation(size_t size)
    : CWiseOperation<SubtractOperation, 2, 1>(size) {}

  auto SubtractOperation::forward(cwise_inputs_t inputs) const -> cwise_outputs_t
  {
    auto [a, b] = inputs;
    return {a - b};
  }

  auto SubtractOperation::backward(cwise_inputs_t inputs, cwise_outputs_t output_gradients) const -> cwise_inputs_t
  {
    auto [gradient] = output_gradients;
    return {gradient, -gradient};
  }
}

