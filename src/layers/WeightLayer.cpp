#include <libkann/layers/WeightLayer.hpp>

#include <iostream>

namespace kann
{
  WeightLayer::WeightLayer(size_t inputSize, size_t outputSize)
    : m_inputSize(inputSize), m_outputSize(outputSize)
  {
    m_weight         = Eigen::MatrixXd::Zero(m_outputSize, m_inputSize);
    m_weightGradient = Eigen::MatrixXd::Zero(m_outputSize, m_inputSize);

    m_bias         = Eigen::VectorXd::Zero(m_outputSize);
    m_biasGradient = Eigen::VectorXd::Zero(m_outputSize);
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

  Eigen::VectorXd WeightLayer::feedForward()
  {
    return m_weight * input() + m_bias;
  }

  Eigen::VectorXd WeightLayer::backPropagate()
  {
    m_weightGradient += outputGradient() * input().transpose();
    m_biasGradient += outputGradient();

    return m_weight.transpose() * outputGradient();
  }

  std::vector<std::span<double>> WeightLayer::params()
  {
    return {
      std::span<double>(m_weight.data(), m_weight.size()),
      std::span<double>(m_bias.data()  , m_bias.size())
    };
  }

  std::vector<std::span<const double>> WeightLayer::params() const
  {
    return {
      std::span<const double>(m_weight.data(), m_weight.size()),
      std::span<const double>(m_bias.data()  , m_bias.size())
    };
  }

  std::vector<std::span<double>> WeightLayer::paramsGradient()
  {
    return {
      std::span<double>(m_weightGradient.data(), m_weightGradient.size()),
      std::span<double>(m_biasGradient.data()  , m_biasGradient.size())
    };
  }

  std::vector<std::span<const double>> WeightLayer::paramsGradient() const
  {
    return {
      std::span<const double>(m_weightGradient.data(), m_weightGradient.size()),
      std::span<const double>(m_biasGradient.data()  , m_biasGradient.size())
    };
  }
}
