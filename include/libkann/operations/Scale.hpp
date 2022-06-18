#pragma once

#include <libkann/Export.hpp>

#include <libkann/Operation.hpp>
#include <libkann/Shape.hpp>

namespace kann
{
  class KANN_EXPORT ScaleOperation : public Operation
  {
  public:
    KANN_EXPORT ScaleOperation(Shape shape, double val);

  public:
    KANN_EXPORT std::vector<Tensor> process(std::vector<Tensor> inputs) const override;
    KANN_EXPORT operation_t differentiate() const override;

  private:
    Shape m_shape;
    double m_val;
  };
}

