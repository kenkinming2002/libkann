#include <libkann/layers/WeightLayer.hpp>

#include <libkann/operations/MatrixMultiplyOperation.hpp>
#include <libkann/operations/ReduceOperation.hpp>

#include <iostream>

namespace kann
{
  WeightLayer::WeightLayer(size_t inputSize, size_t outputSize)
    : m_inputSize(inputSize), m_outputSize(outputSize)
  {
    m_weight = std::make_shared<const Tensor>(m_inputSize * m_outputSize);
    m_bias   = std::make_shared<const Tensor>(m_outputSize);
  }

  // Do not copy variable
  WeightLayer::WeightLayer(const WeightLayer& other)
    : m_inputSize(other.m_inputSize), m_outputSize(other.m_outputSize),
      m_weight(other.m_weight), m_bias(other.m_bias) {}

  std::unique_ptr<Layer> WeightLayer::clone() const
  {
    return std::make_unique<WeightLayer>(*this);
  }

  size_t WeightLayer::inputSize() const
  {
    return m_inputSize;
  }

  size_t WeightLayer::outputSize() const
  {
    return m_outputSize;
  }

  std::vector<std::shared_ptr<const Variable>> WeightLayer::parametersVariables() const
  {
    return {m_weightVariable, m_biasVariable};
  }

  std::vector<std::shared_ptr<const Tensor>> WeightLayer::parameters() const
  {
    return {m_weight, m_bias};
  }

  void WeightLayer::parameters(std::vector<std::shared_ptr<const Tensor>> parameters)
  {
    assert(parameters.size() == 2);
    m_weight = std::move(parameters[0]);
    m_bias   = std::move(parameters[1]);
  }

  auto WeightLayer::operator()(std::shared_ptr<const Variable> input, StateVariables state) const -> std::pair<std::shared_ptr<const Variable>, StateVariables>
  {
    // TODO: Fuse them into a single operation
    auto prod = std::make_shared<const Variable>(std::vector{m_weightVariable, std::move(input)}, std::make_shared<MatrixMultiplyOperation>(m_outputSize, 1, m_inputSize, false, false));
    auto output =  std::make_shared<const Variable>(std::vector{m_biasVariable, std::move(prod)}, std::make_shared<ReduceOperation>(2));
    return std::make_pair(std::move(output), std::move(state));
  }

}
