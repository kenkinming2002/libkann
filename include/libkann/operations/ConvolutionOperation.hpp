#pragma once

#include <libkann/Operation.hpp>
#include <libkann/Vec.hpp>

namespace kann
{
  // This should be cross-correlation
  class CrossCorrelationOperation : public OperationImpl<CrossCorrelationOperation, 2, 1>
  {
  public:
    CrossCorrelationOperation(Vec2 input_size, Vec2 output_size, Vec2 kernel_size);

  public:
    outputs_t process_impl(inputs_t inputs) const;
    operation_t differentiate() const override;

  private:
    Vec2 m_input_size, m_output_size, m_kernel_size;
  };
}

