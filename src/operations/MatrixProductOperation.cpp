#include <libkann/operations/MatrixProductOperation.hpp>

namespace kann
{
  MatrixProductOperation::MatrixProductOperation(size_t m, size_t n, size_t k)
    : m_m(m), m_n(n), m_k(k) {}

  auto MatrixProductOperation::process_impl(inputs_t inputs) const -> outputs_t
  {
    // a: m_m * m_k
    // b: m_k * m_n
    const auto& [a, b] = inputs;

    Tensor output(m_m * m_n);
    if(m_k == 1)
      output.asMatrix(m_m,m_n).noalias() = a->asVector() * b->asRowVector();
    else
      output.asMatrix(m_m,m_n).noalias() = a->asMatrix(m_m, m_k) * b->asMatrix(m_k, m_n);

    return { std::move(output) };
  }

  class MatrixProductGradientOperation : public OperationImpl<MatrixProductGradientOperation, 3, 2>
  {
  public:
    MatrixProductGradientOperation(size_t m, size_t n, size_t k)
    : m_m(m), m_n(n), m_k(k) {}

  public:
    outputs_t process_impl(inputs_t inputs) const
    {
      // a: m_m * m_k
      // b: m_k * m_n
      // output_gradient : m_m * m_n
      const auto& [a, b, output_gradient] = inputs;

      Tensor gradient_a(m_m * m_k);
      Tensor gradient_b(m_k * m_n);

      if(m_n == 1)
        gradient_a.asMatrix(m_m, m_k) = output_gradient->asVector() * b->asRowVector();
      else
        gradient_a.asMatrix(m_m, m_k) = output_gradient->asMatrix(m_m, m_n) * b->asMatrix(m_k, m_n).transpose();

      if(m_m == 1)
        gradient_b.asMatrix(m_k, m_n) = a->asVector() * output_gradient->asRowVector();
      else
        gradient_b.asMatrix(m_k, m_n) = a->asMatrix(m_m, m_k).transpose() * output_gradient->asMatrix(m_m, m_n);

      return { std::move(gradient_a), std::move(gradient_b) };
    }

  private:
    size_t m_m, m_n, m_k;
  };

  operation_t MatrixProductOperation::differentiate() const
  {
    return std::make_shared<MatrixProductGradientOperation>(m_m, m_n, m_k);
  }
}

