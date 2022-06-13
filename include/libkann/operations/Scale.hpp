#pragma once

#include <libkann/Operation.hpp>

namespace kann
{
  class ScaleOperation : public Operation
  {
  public:
    ScaleOperation(size_t size, double val);

  public:
    std::vector<tensor_t> process(std::vector<tensor_t> inputs) const override;
    operation_t differentiate() const override;

  private:
    size_t m_size;
    double m_val;
  };
}

