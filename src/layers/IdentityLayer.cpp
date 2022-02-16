#include <libkann/layers/IdentityLayer.hpp>

#include <libkann/operations/IdentityOperation.hpp>

namespace kann
{
  using namespace std::placeholders;

  IdentityLayer::IdentityLayer(size_t inputSize, size_t outputSize, size_t offset)
    : m_inputSize(inputSize), m_outputSize(outputSize), m_offset(offset) {}

  size_t IdentityLayer::inputSize() const
  {
    return m_inputSize;
  }

  size_t IdentityLayer::outputSize() const
  {
    return m_outputSize;
  }

  Layer::Output IdentityLayer::process(Scope scope, Input input) const
  {
    // Fast path
    auto inputVariable = std::move(input.input);

    auto outputVariable = m_inputSize == m_outputSize
      ? std::move(inputVariable)
      : std::make_shared<const Variable>(
         std::vector{std::move(inputVariable)},
         std::make_shared<IdentityOperation>(m_inputSize, m_outputSize, m_offset)
        );

    return Output{
      std::move(outputVariable),
      std::move(input.inputState)
    };
  }
}

