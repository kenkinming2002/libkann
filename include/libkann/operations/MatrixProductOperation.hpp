#pragma once

#include <libkann/Operation.hpp>

namespace kann
{
  class MatrixProductOperation : public OperationImpl<MatrixProductOperation, 2>
  {
  public:
    /* Mutiply a mxk matrix with a kxn matrix.
     *
     * @param transpose 1 whether a first matrix(matrix with dimension mxk)
     *        should be transposed
     * @param transpose 1 whether a second matrix(matrix with dimension kxn)
     *        should be transposed */
    MatrixProductOperation(size_t m, size_t n, size_t k, bool transpose1, bool transpose2);

  public:
    Tensor process_impl(inputs_t inputs) const;
    variables_t gradients_impl(variable_t gradient, variables_t inputs) const;

  private:
    size_t m_m, m_n, m_k;
    bool m_transpose1, m_transpose2;
  };
}
