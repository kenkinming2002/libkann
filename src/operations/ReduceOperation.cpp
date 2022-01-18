#include <libkann/operations/ReduceOperation.hpp>

#include <libkann/Variable.hpp>

namespace kann
{
  ReduceOperation::ReduceOperation(size_t inputCount)
    : m_inputCount(inputCount)
  {
    assert(m_inputCount != 0);
  }

  std::shared_ptr<const Tensor> ReduceOperation::process(std::vector<std::shared_ptr<const Tensor>> inputs) const
  {
    assert(inputs.size() == m_inputCount);

    auto result = std::make_shared<Tensor>(inputs.front()->size());
    result->asArray().setZero();
    for(auto& input : inputs)
      result->asArray() += input->asArray();

    return result;
  }

  VariableList ReduceOperation::gradients(VariableHandle gradient, VariableList /*inputs*/) const
  {
    return std::vector(m_inputCount, gradient);
  }
}
