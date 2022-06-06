#include <libkann/operations/ReduceOperation.hpp>

#include <range/v3/all.hpp>

namespace kann
{
  ReduceOperation::ReduceOperation(size_t size, size_t count)
    : m_size(size), m_count(count) {}

  std::vector<tensor_t> ReduceOperation::process(std::vector<tensor_t> inputs) const
  {
    assert(inputs.size() == m_count);
    assert(ranges::all_of(inputs, [this](const tensor_t& input) { return input->size() == m_size; }));

    Tensor result = [&]()
    {
      if(m_count == 0)
        return Tensor::constant(m_size, 0.0);

      Tensor result(m_size);
      result.asArray() = inputs.front()->asArray();
      for(const tensor_t& input : inputs | ranges::views::drop(1))
        result.asArray() += input->asArray();

      return result;
    }();


    return { std::make_shared<const Tensor>(std::move(result)) };
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
      assert(ranges::all_of(inputs, [this](const tensor_t& input) { return input->size() == m_size; }));

      // Note: First m_count elements of inputs are original inputs while the last element is output_gradient
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
