#include <libkann/operations/ReduceOperation.hpp>

#include <range/v3/all.hpp>

namespace kann
{
  ReduceOperation::ReduceOperation(size_t size, size_t count)
    : m_size(size), m_count(count) {}

  std::vector<tensor_t> ReduceOperation::process(std::vector<tensor_t> inputs) const
  {
    assert(inputs.size() == m_count);
    Tensor output = Tensor::reduce(inputs
        | ranges::views::transform([](tensor_t& input) { return input.get(); })
        | ranges::to_vector);
    return { std::make_shared<const Tensor>(std::move(output)) };
  }

  class ReduceOperationGradient : public Operation
  {
  public:
    ReduceOperationGradient(size_t size, size_t count)
      : m_size(size), m_count(count) {}

  public:
    std::vector<tensor_t> process(std::vector<tensor_t> inputs) const override
    {
      assert(inputs.size() == m_count + 1);
      return std::vector(m_count, inputs.back());
    }

  private:
    size_t m_size;
    size_t m_count;
  };


  operation_t ReduceOperation::differentiate() const
  {
    return std::make_shared<ReduceOperationGradient>(m_size, m_count);
  }
}
