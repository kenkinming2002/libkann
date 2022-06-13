#include <libkann/operations/ConcatOperation.hpp>

#include <libkann/Tensor.hpp>

#include <range/v3/all.hpp>

namespace kann
{
  ConcatOperation::ConcatOperation(size_t size, size_t count)
    : m_size(size), m_count(count) {}

  std::vector<tensor_t> ConcatOperation::process(std::vector<tensor_t> op_inputs) const
  {
    assert(op_inputs.size() == m_count);
    std::vector<const Tensor*> inputs = op_inputs
      | ranges::views::transform([](const tensor_t& input) { return input.get(); })
      | ranges::to_vector;

    Tensor output = Tensor::concat(std::move(inputs), m_size, m_count);
    return {std::make_shared<const Tensor>(std::move(output))};
  }

  class ConcatGradientOperation : public Operation
  {
  public:
    ConcatGradientOperation(size_t size, size_t count)
      : m_size(size), m_count(count) {}

  public:
    std::vector<tensor_t> process(std::vector<tensor_t> op_inputs) const override
    {
      assert(op_inputs.size() == m_count + 1);
      const Tensor& output_gradient = *op_inputs.back();

      std::vector<Tensor> input_gradients = Tensor::split(output_gradient, m_size, m_count);
      return input_gradients
        | ranges::views::transform([](Tensor& output) { return std::make_shared<const Tensor>(std::move(output)); })
        | ranges::to_vector;
    }

  private:
    size_t m_size, m_count;
  };


  operation_t ConcatOperation::differentiate() const
  {
    return std::make_shared<ConcatGradientOperation>(m_size, m_count);
  }
}
