#include <libkann/operations/SplitOperation.hpp>

namespace kann
{
  SplitOperation::SplitOperation(size_t size, size_t count)
    : m_size(size), m_count(count) {}

  std::vector<tensor_t> SplitOperation::process(std::vector<tensor_t> op_inputs) const
  {
    assert(op_inputs.size() == 1);
    const Tensor& input = *op_inputs[0];

    std::vector<Tensor> outputs = Tensor::split(input, m_size, m_count);
    return outputs
      | ranges::views::transform([](Tensor& output) { return std::make_shared<const Tensor>(std::move(output)); })
      | ranges::to_vector;
  }

  class SplitGradientOperation : public Operation
  {
  public:
    SplitGradientOperation(size_t size, size_t count)
      : m_size(size), m_count(count) {}

  public:
    std::vector<tensor_t> process(std::vector<tensor_t> op_inputs) const override
    {
      // Concat all but the first input, which is gradient
      assert(op_inputs.size() == 1 + m_count);
      std::vector<const Tensor*> output_gradients = op_inputs
        | ranges::views::drop_exactly(1)
        | ranges::views::transform([](const tensor_t& v) { return v.get(); })
        | ranges::to_vector;

      Tensor input_gradient = Tensor::concat(std::move(output_gradients), m_size, m_count);
      return {std::make_shared<const Tensor>(std::move(input_gradient))};
    }

  private:
    size_t m_size, m_count;
  };

  operation_t SplitOperation::differentiate() const
  {
    return std::make_shared<SplitGradientOperation>(m_size, m_count);
  }
}
