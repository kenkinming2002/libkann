#pragma once

#include <libkann/Operation.hpp>
#include <libkann/Vec.hpp>

namespace kann
{
  // This should be cross-correlation
  class CrossCorrelationOperation : public Operation
  {
  public:
    CrossCorrelationOperation(Vec2 kernel_size, Vec2 padding_size);

  public:
    std::vector<Tensor> process(std::vector<Tensor> inputs) const override;
    operation_t differentiate() const override;

  private:
    Vec2 m_kernel_size;
    Vec2 m_padding_size;
  };
}

