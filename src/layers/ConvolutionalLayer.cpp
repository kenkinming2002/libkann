#include <libkann/layers/ConvolutionalLayer.hpp>

#include <assert.h>

namespace kann
{
  ConvolutionalLayer::ConvolutionalLayer(size_t inputWidth, size_t inputHeight, size_t kernelSize, size_t inputChannelCount, size_t outputChannelCount)
    : m_inputWidth(inputWidth), m_inputHeight(inputHeight),
      m_kernelSize(kernelSize),
      m_inputChannelCount(inputChannelCount), m_outputChannelCount(outputChannelCount)
  {
    /* Unsigned overflow is well-defined behavior so we could safely do the check
     * afterwards */
    assert(m_inputWidth  >= m_kernelSize);
    assert(m_inputHeight >= m_kernelSize);

    m_kernels.resize(m_inputChannelCount * m_outputChannelCount, Eigen::MatrixXd::Zero(m_kernelSize, m_kernelSize));
    m_kernelsGradient.resize(m_inputChannelCount * m_outputChannelCount, Eigen::MatrixXd::Zero(m_kernelSize, m_kernelSize));
  }

  std::unique_ptr<Layer> ConvolutionalLayer::clone() const
  {
    return std::make_unique<ConvolutionalLayer>(*this);
  }

  size_t ConvolutionalLayer::inputSize() const
  {
    return m_inputWidth * m_inputHeight * m_inputChannelCount;
  }

  size_t ConvolutionalLayer::outputSize() const
  {
    return (m_inputWidth-m_kernelSize+1) * (m_inputHeight-m_kernelSize+1) * m_outputChannelCount;
  }

  auto ConvolutionalLayer::operator()(std::shared_ptr<const Variable> input, StateVariables state) const -> std::pair<std::shared_ptr<const Variable>, StateVariables>
  {
    assert(false && "Unimplemented");
  }
}
