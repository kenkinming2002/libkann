#include <libkann/operations/CopyOperation.hpp>

#include <libkann/Tensor.hpp>

#include <range/v3/all.hpp>

namespace kann
{
  CopyOperation::CopyOperation(size_t size, size_t count)
    : m_size(size), m_count(count) {}

  std::vector<tensor_t> CopyOperation::process(std::vector<tensor_t> inputs) const
  {
    assert(inputs.size() == 1);
    assert(inputs.front()->size() == m_size);
    return std::vector(m_count, inputs.front());
  }

  class CopyOperationGradient : public Operation
  {
  public:
    CopyOperationGradient(size_t size, size_t count)
      : m_size(size), m_count(count) {}

  public:
    std::vector<tensor_t> process(std::vector<tensor_t> inputs) const override
    {
      assert(inputs.size() == 1 + m_count);
      Tensor input_gradient = Tensor::reduce(inputs
        | ranges::views::drop_exactly(1)
        | ranges::views::transform([](tensor_t& input) { return input.get(); })
        | ranges::to_vector);
      return { std::make_shared<const Tensor>(std::move(input_gradient)) };
    }

  private:
    size_t m_size;
    size_t m_count;
  };

  operation_t CopyOperation::differentiate() const
  {
    return std::make_shared<CopyOperationGradient>(m_size, m_count);
  }
}
