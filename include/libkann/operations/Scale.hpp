#pragma once

#include <libkann/Operation.hpp>
#include <libkann/Shape.hpp>

namespace kann
{
  class ScaleOperation : public Operation
  {
  public:
    ScaleOperation(Shape shape, double val);

  public:
    std::vector<Tensor> process(std::vector<Tensor> inputs) const override;
    operation_t differentiate() const override;

  private:
    Shape m_shape;
    double m_val;
  };
}

