#include <libkann/operations/MatrixMultiplyOperation.hpp>

#include <libkann/Variable.hpp>

namespace kann
{
  MatrixMultiplyOperation::MatrixMultiplyOperation(size_t m, size_t n, size_t k, bool transpose1, bool transpose2)
    : m_m(m), m_n(n), m_k(k), m_transpose1(transpose1), m_transpose2(transpose2) {}

  Tensor MatrixMultiplyOperation::processImpl(const Tensor& a, const Tensor& b) const
  {
    Tensor result(m_m * m_n);
    if(m_n == 1)
    {
      // This is actually just a matrix vector multiplication
      if(m_transpose1)
        result.asVector().noalias() = a.asMatrix(m_k, m_m).transpose() * b.asVector();
      else
        result.asVector().noalias() = a.asMatrix(m_m, m_k) * b.asVector();
    }
    else if(m_m == 1)
    {
      // This is also just a matrix vector multiplication
      if(m_transpose2)
        result.asRowVector().noalias() = a.asRowVector() * b.asMatrix(m_n, m_k).transpose();
      else
        result.asRowVector().noalias() = a.asRowVector() * b.asMatrix(m_k, m_n);
    }
    else if(m_k == 1)
    {
      // a is a mx1 matrix, b is a 1xn matrix
      result.asMatrix(m_m,m_n).noalias() = a.asVector() * b.asRowVector();
    }
    else
    {
      if(m_transpose1)
      {
        if(m_transpose2)
          result.asMatrix(m_m,m_n).noalias() = a.asMatrix(m_k, m_m).transpose() * b.asMatrix(m_n, m_k).transpose();
        else
          result.asMatrix(m_m,m_n).noalias() = a.asMatrix(m_k, m_m).transpose() * b.asMatrix(m_k, m_n);
      }
      else
      {
        if(m_transpose2)
          result.asMatrix(m_m,m_n).noalias() = a.asMatrix(m_m, m_k) * b.asMatrix(m_n, m_k).transpose();
        else
          result.asMatrix(m_m,m_n).noalias() = a.asMatrix(m_m, m_k) * b.asMatrix(m_k, m_n);
      }
    }

    return result;
  }

  VariablePair MatrixMultiplyOperation::gradientsImpl(VariableHandle gradient, VariableHandle a, VariableHandle b) const
  {
    if(m_transpose1 || m_transpose2)
      throw std::runtime_error("Not Implemented");

    /* Note:
     *  variable gradient refers to a mxn matrix
     *  variable a refers to a mxk matrix
     *  variable b refers to a kxn matrix
     */
    return std::make_pair(
      std::make_shared<const Variable>(std::vector{gradient, b}, std::make_shared<MatrixMultiplyOperation>(m_m, m_k, m_n, false, true)),
      std::make_shared<const Variable>(std::vector{a, gradient}, std::make_shared<MatrixMultiplyOperation>(m_k, m_n, m_m, true, false))
    );
  }
}

