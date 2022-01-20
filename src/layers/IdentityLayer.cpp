#include <libkann/layers/IdentityLayer.hpp>

#include <libkann/operations/IdentityOperation.hpp>

namespace kann
{
  using namespace std::placeholders;

  IdentityLayer::IdentityLayer(size_t inputSize, size_t outputSize, size_t offset)
    : m_inputSize(inputSize), m_outputSize(outputSize), m_offset(offset) {}

  std::unique_ptr<Layer> IdentityLayer::clone() const
  {
    return std::make_unique<IdentityLayer>(*this);
  }

  size_t IdentityLayer::inputSize() const
  {
    return m_inputSize;
  }

  size_t IdentityLayer::outputSize() const
  {
    return m_outputSize;
  }

  std::vector<std::shared_ptr<const Variable>> IdentityLayer::parametersVariables(unsigned tags) const
  {
    return {};
  }

  std::vector<std::reference_wrapper<const std::shared_ptr<const Tensor>>> IdentityLayer::parameters(unsigned tags) const
  {
    return {};
  }

  std::vector<std::reference_wrapper<std::shared_ptr<const Tensor>>> IdentityLayer::parameters(unsigned tags)
  {
    return {};
  }

  auto IdentityLayer::operator()(std::shared_ptr<const Variable> input, StateVariables state) const -> std::pair<std::shared_ptr<const Variable>, StateVariables>
  {
    if(m_inputSize == m_outputSize)
    {
      // Fast path
      auto output = std::move(input);
      return std::make_pair(std::move(output), std::move(state));
    }
    else
    {
      auto output = std::make_shared<const Variable>(std::vector{std::move(input)}, std::make_shared<IdentityOperation>(m_inputSize, m_outputSize, m_offset));
      return std::make_pair(std::move(output), std::move(state));
    }
  }
}

