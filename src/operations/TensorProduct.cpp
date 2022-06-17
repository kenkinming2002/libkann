#include <libkann/operations/TensorProduct.hpp>

#include <libkann/operations/Impl.hpp>

namespace kann
{
  TensorProductOperation::TensorProductOperation(size_t rank_m, size_t rank_n, size_t rank_k)
    : m_rank_m(rank_m), m_rank_n(rank_n), m_rank_k(rank_k) {}

  std::vector<Tensor> TensorProductOperation::process(std::vector<Tensor> inputs) const
  {
    return operation_process_impl<2, 1>(std::move(inputs), [this](const Tensor& a, const Tensor& b) {
      return std::make_tuple(math::product(a, b,
        m_rank_m, m_rank_n, m_rank_k,
        false, false));
    });
  }

  class TensorProductGradientOperation : public Operation
  {
  public:
    TensorProductGradientOperation(size_t rank_m, size_t rank_n, size_t rank_k)
      : m_rank_m(rank_m), m_rank_n(rank_n), m_rank_k(rank_k) {}

  public:
    std::vector<Tensor> process(std::vector<Tensor> inputs) const override
    {
      return operation_process_impl<3, 2>(std::move(inputs), [this](const Tensor& a, const Tensor& b, const Tensor& output_gradient) {
        return std::make_tuple(
          math::product(output_gradient, b, m_rank_m, m_rank_k, m_rank_n, false, true),
          math::product(a, output_gradient, m_rank_k, m_rank_n, m_rank_m, true, false));
      });
    }

  private:
    size_t m_rank_m, m_rank_n, m_rank_k;
  };

  operation_t TensorProductOperation::differentiate() const
  {
    return std::make_shared<TensorProductGradientOperation>(m_rank_m, m_rank_n, m_rank_k);
  }
}

