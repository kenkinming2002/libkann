#include <libkann/operations/ReduceOperation.hpp>

#include <libkann/Variable.hpp>

namespace kann
{
  ReduceOperation::ReduceOperation(size_t inputCount)
    : m_inputCount(inputCount)
  {
    assert(m_inputCount != 0);
  }

  tensor_t ReduceOperation::process(std::vector<const Tensor*> inputs) const
  {
    assert(inputs.size() == m_inputCount);

    auto result = std::make_shared<Tensor>(inputs.front()->size());
    switch(m_inputCount)
    {
    case 1:
      result->asArray() = inputs.front()->asArray();
      break;
    default:
      result->asArray() = inputs[0]->asArray() + inputs[1]->asArray();
      for(size_t i=2; i<inputs.size(); ++i)
        result->asArray() += inputs[i]->asArray();
      break;
    }
    return result;
  }

  std::vector<variable_t> ReduceOperation::gradients(variable_t gradient, std::vector<variable_t> /*inputs*/) const
  {
    return std::vector(m_inputCount, gradient);
  }
}
