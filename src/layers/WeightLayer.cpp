#include <libkann/layers/WeightLayer.hpp>

#include <libkann/operations/MatrixMultiplyOperation.hpp>
#include <libkann/operations/ReduceOperation.hpp>

#include <iostream>

namespace kann
{
  WeightLayer::WeightLayer(size_t inputSize, size_t outputSize)
    : m_inputSize(inputSize), m_outputSize(outputSize)
  {
    m_weight = std::make_shared<Parameter>(m_inputSize * m_outputSize);
    m_bias   = std::make_shared<Parameter>(m_outputSize);
  }

  WeightLayer::WeightLayer(const WeightLayer& other)
  {
    *this = other;
  }

  WeightLayer& WeightLayer::operator=(const WeightLayer& other)
  {
    Layer::operator=(other);

    m_inputSize  = other.m_inputSize;
    m_outputSize = other.m_outputSize;
    m_weight = std::make_shared<Parameter>(*other.m_weight);
    m_bias   = std::make_shared<Parameter>(*other.m_bias);

    return *this;
  }

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

  std::vector<std::shared_ptr<const Parameter>> WeightLayer::parameters(unsigned tags) const
  {
    return {m_weight, m_bias};
  }

  std::vector<std::shared_ptr<Parameter>> WeightLayer::parameters(unsigned tags)
  {
    return {m_weight, m_bias};
  }


  auto WeightLayer::operator()(std::shared_ptr<const Variable> input, StateVariables state) const -> std::pair<std::shared_ptr<const Variable>, StateVariables>
  {
    // TODO: Fuse them into a single operation
    auto prod = std::make_shared<const Variable>(std::vector{m_weight->variable, std::move(input)}, std::make_shared<MatrixMultiplyOperation>(m_outputSize, 1, m_inputSize, false, false));
    auto output =  std::make_shared<const Variable>(std::vector{m_bias->variable, std::move(prod)}, std::make_shared<ReduceOperation>(2));
    return std::make_pair(std::move(output), std::move(state));
  }

}
