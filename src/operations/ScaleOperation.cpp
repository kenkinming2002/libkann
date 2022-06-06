#include <libkann/operations/ScaleOperation.hpp>

namespace kann
{
  ScaleOperation::ScaleOperation(size_t size, double val)
    : CWiseOperation<ScaleOperation, 1, 1>(size), m_val(val) {}

  auto ScaleOperation::forward(cwise_inputs_t inputs) const -> cwise_outputs_t
  {
    auto [input] = inputs;
    return {input * m_val};
  }

  auto ScaleOperation::backward(cwise_inputs_t inputs, cwise_outputs_t output_gradients) const -> cwise_inputs_t
  {
    auto [gradient] = output_gradients;
    return {gradient * m_val};
  }
}
