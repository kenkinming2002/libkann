#pragma once

#include <libkann/Operation.hpp>

namespace kann
{
  class MatrixProductOperation : public OperationImpl<MatrixProductOperation, 2, 1>
  {
  public:
    /* Mutiply a mxk matrix with a kxn matrix.*/
    MatrixProductOperation(size_t m, size_t n, size_t k);

  public:
    outputs_t process_impl(inputs_t inputs) const;
    operation_t differentiate() const override;

  private:
    size_t m_m, m_n, m_k;
  };
}
