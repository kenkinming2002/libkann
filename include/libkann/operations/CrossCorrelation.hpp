#pragma once

#include <libkann/Export.hpp>

#include <libkann/Operation.hpp>
#include <libkann/Vec.hpp>

namespace kann
{
  // This should be cross-correlation
  class KANN_EXPORT CrossCorrelationOperation : public Operation
  {
  public:
    KANN_EXPORT CrossCorrelationOperation(Vec2 kernel_size, Vec2 padding_size);

  public:
    KANN_EXPORT std::vector<Tensor> process(std::vector<Tensor> inputs) const override;
    KANN_EXPORT operation_t differentiate() const override;

  private:
    Vec2 m_kernel_size;
    Vec2 m_padding_size;
  };
}

