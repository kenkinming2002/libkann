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

  Layer::ProcessOutput IdentityLayer::process(ProcessInput input) const
  {
    ProcessOutput output;
    output.variable = m_inputSize == m_outputSize
      ? std::move(input.variable)
      : Variable::apply(IdentityOperation(m_inputSize, m_outputSize, m_offset), {std::move(input.variable)});

    output.states = std::move(input.states);
    return output;
  }
}

