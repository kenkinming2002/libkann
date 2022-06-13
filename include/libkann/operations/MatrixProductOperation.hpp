#pragma once

#include <libkann/Operation.hpp>

namespace kann
{
  class MatrixProductOperation : public Operation
  {
  public:
    /* Mutiply a mxk matrix with a kxn matrix.*/
    MatrixProductOperation(size_t m, size_t n, size_t k);

  public:
    std::vector<tensor_t> process(std::vector<tensor_t> inputs) const override;
    operation_t differentiate() const override;

  private:
    size_t m_m, m_n, m_k;
  };
}
