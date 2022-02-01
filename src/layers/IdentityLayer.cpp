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

  LayerVariable IdentityLayer::operator()(Scope scope, LayerVariable input) const
  {
    // Fast path
    if(m_inputSize == m_outputSize)
      return input;

    auto inputVariable = std::move(input.variable);
    auto outputVariable = std::make_shared<const Variable>(
      std::vector{inputVariable},
      std::make_shared<IdentityOperation>(m_inputSize, m_outputSize, m_offset)
    );

    auto output = std::move(input);
    output.variable = std::move(outputVariable);
    return output;
  }
}

