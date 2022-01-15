#include <libkann/functions/ReduceFunction.hpp>

#include <libkann/operations/ReduceOperation.hpp>

namespace kann
{
  ReduceFunction::ReduceFunction(size_t inputCount)
    : m_inputCount(inputCount) {}

  std::shared_ptr<const Variable> ReduceFunction::impl(std::shared_ptr<const Variable> v) const
  {
    const auto reduceOperation = std::make_shared<ReduceOperation>(m_inputCount);
    return std::make_shared<const Variable>(std::vector{v}, reduceOperation);
  }
}

