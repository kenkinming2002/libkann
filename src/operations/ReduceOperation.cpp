#include <libkann/operations/ReduceOperation.hpp>

#include <libkann/functions/ReduceFunction.hpp>

namespace kann
{
  ReduceOperation::ReduceOperation(size_t inputCount)
    : m_inputCount(inputCount)
  {
    assert(m_inputCount != 0);
  }

  Tensor ReduceOperation::process(const std::vector<Tensor>& inputs) const
  {
    assert(inputs.size() == m_inputCount);
    Tensor result(inputs.front().size());
    result.asArray().setZero();
    for(const auto& input : inputs)
      result.asArray() += input.asArray();

    return result;
  }

  std::shared_ptr<const Function> ReduceOperation::function() const
  {
    const auto reduceFunction = std::make_shared<ReduceFunction>(m_inputCount);
    return reduceFunction;
  }
}
