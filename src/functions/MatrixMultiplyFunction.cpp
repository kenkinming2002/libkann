#include <libkann/functions/MatrixMultiplyFunction.hpp>

#include <libkann/operations/MatrixMultiplyOperation.hpp>

namespace kann
{
  MatrixMultiplyFunction::MatrixMultiplyFunction(size_t m, size_t n, size_t k, bool transpose1, bool transpose2)
    : m_m(m), m_n(n), m_k(k), m_transpose1(transpose1), m_transpose2(transpose2) {}

  std::shared_ptr<const Variable> MatrixMultiplyFunction::impl(std::shared_ptr<const Variable> a, std::shared_ptr<const Variable> b) const
  {
    const auto matrixMultiplyOperation = std::make_shared<MatrixMultiplyOperation>(m_m, m_n, m_k, m_transpose1, m_transpose2);
    return std::make_shared<const Variable>(std::vector{a, b}, matrixMultiplyOperation);
  }
}
