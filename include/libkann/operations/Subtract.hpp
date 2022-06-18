#pragma once

#include <libkann/Export.hpp>

#include <libkann/Operation.hpp>

namespace kann
{
  class KANN_EXPORT SubtractOperation : public Operation
  {
  public:
    KANN_EXPORT SubtractOperation(Shape shape);

  public:
    KANN_EXPORT std::vector<Tensor> process(std::vector<Tensor> inputs) const override;
    KANN_EXPORT operation_t differentiate() const override;

  private:
    Shape m_shape;
  };
}


