#include <libkann/layers/IdentityLayer.hpp>

namespace kann
{
  using namespace std::placeholders;

  IdentityLayer::IdentityLayer(size_t inputSize, size_t outputSize, size_t offset)
    : Layer(0), m_inputSize(inputSize), m_outputSize(outputSize), m_offset(offset) {}

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

  void IdentityLayer::feedForward(const Eigen::VectorXd& input, Eigen::VectorXd& output) const
  {
    output = Eigen::VectorXd::Zero(m_outputSize);

    if(m_inputSize >= m_outputSize)
      output = input.segment(m_offset, m_outputSize);
    else
      output.segment(m_offset, m_inputSize) = input;
  }

  void IdentityLayer::backPropagate(const Eigen::VectorXd& input, const Eigen::RowVectorXd& outputGradient, Eigen::RowVectorXd& inputGradient, Eigen::ArrayXd& layerGradient) const
  {
    inputGradient = Eigen::RowVectorXd::Zero(m_inputSize);
    if(m_inputSize >= m_outputSize)
      inputGradient.segment(m_offset, m_outputSize) = outputGradient;
    else
      inputGradient = outputGradient.segment(m_offset, m_inputSize);

    layerGradient.resize(0); // We do not have any gradient
  }
}

