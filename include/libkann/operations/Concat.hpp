#pragma once

#include <libkann/Operation.hpp>

namespace kann
{
  class ConcatOperation : public Operation
  {
  public:
    ConcatOperation(size_t size, size_t count);

  public:
    std::vector<tensor_t> process(std::vector<tensor_t> inputs) const override;
    operation_t differentiate() const override;

  private:
    size_t m_size, m_count;
  };
}

