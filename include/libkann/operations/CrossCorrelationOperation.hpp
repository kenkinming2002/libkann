#pragma once

#include <libkann/Operation.hpp>
#include <libkann/Vec.hpp>

namespace kann
{
  // This should be cross-correlation
  class CrossCorrelationOperation : public Operation
  {
  public:
    CrossCorrelationOperation(Vec2 input_size, Vec2 output_size, Vec2 kernel_size);

  public:
    std::vector<tensor_t> process(std::vector<tensor_t> inputs) const override;
    operation_t differentiate() const override;

  private:
    Vec2 m_input_size, m_output_size, m_kernel_size;
  };
}

