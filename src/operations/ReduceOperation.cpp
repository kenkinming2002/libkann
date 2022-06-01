#include <libkann/operations/ReduceOperation.hpp>

#include <libkann/Variable.hpp>
#include <range/v3/all.hpp>

namespace kann
{
  ReduceOperation::ReduceOperation(size_t input_count)
    : m_input_count(input_count)
  {
    assert(m_input_count != 0);
  }

  tensor_t ReduceOperation::process(std::vector<const Tensor*> inputs) const
  {
    Tensor result(inputs.front()->size());
    result.asArray() = inputs.front()->asArray();
    for(const Tensor* input : inputs | ranges::views::drop(1))
      result.asArray() += input->asArray();

    return std::make_shared<const Tensor>(std::move(result));
  }

  std::vector<variable_t> ReduceOperation::gradients(variable_t gradient, std::vector<variable_t> /*inputs*/) const
  {
    return ranges::views::repeat_n(gradient, m_input_count) | ranges::to_vector;
  }
}
