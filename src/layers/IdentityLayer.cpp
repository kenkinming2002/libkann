#include <libkann/layers/IdentityLayer.hpp>

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

  Eigen::VectorXd IdentityLayer::feedForward()
  {
    Eigen::VectorXd output = Eigen::VectorXd::Zero(m_outputSize);
    if(m_inputSize >= m_outputSize)
      output = input().segment(m_offset, m_outputSize);
    else
      output.segment(m_offset, m_inputSize) = input();

    return output;
  }

  Eigen::VectorXd IdentityLayer::backPropagate()
  {
    Eigen::VectorXd inputGradient = Eigen::RowVectorXd::Zero(m_inputSize);
    if(m_inputSize >= m_outputSize)
      inputGradient.segment(m_offset, m_outputSize) = outputGradient();
    else
      inputGradient = outputGradient().segment(m_offset, m_inputSize);

    return inputGradient;
  }

  std::vector<std::span<double>> IdentityLayer::params()
  {
    return {};
  }

  std::vector<std::span<const double>> IdentityLayer::params() const
  {
    return {};
  }

  std::vector<std::span<double>> IdentityLayer::paramsGradient()
  {
    return {};
  }

  std::vector<std::span<const double>> IdentityLayer::paramsGradient() const
  {
    return {};
  }
}

