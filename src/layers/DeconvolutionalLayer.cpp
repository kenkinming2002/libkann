#include <libkann/layers/DeconvolutionalLayer.hpp>

#include <assert.h>

namespace kann
{
  DeconvolutionalLayer::DeconvolutionalLayer(size_t inputWidth, size_t inputHeight, size_t kernelSize, size_t inputChannelCount, size_t outputChannelCount)
    : m_inputWidth(inputWidth), m_inputHeight(inputHeight),
      m_kernelSize(kernelSize),
      m_inputChannelCount(inputChannelCount), m_outputChannelCount(outputChannelCount)
  {
    m_kernels.resize(m_inputChannelCount * m_outputChannelCount, Eigen::MatrixXd::Zero(m_kernelSize, m_kernelSize));
    m_kernelsGradient.resize(m_inputChannelCount * m_outputChannelCount, Eigen::MatrixXd::Zero(m_kernelSize, m_kernelSize));
  }

  std::unique_ptr<Layer> DeconvolutionalLayer::clone() const
  {
    return std::make_unique<DeconvolutionalLayer>(*this);
  }

  size_t DeconvolutionalLayer::inputSize() const
  {
    return m_inputWidth * m_inputHeight * m_inputChannelCount;
  }

  size_t DeconvolutionalLayer::outputSize() const
  {
    return (m_inputWidth+m_kernelSize-1) * (m_inputHeight+m_kernelSize-1) * m_outputChannelCount;
  }

  auto DeconvolutionalLayer::operator()(std::shared_ptr<const Variable> input, StateVariables state) const -> std::pair<std::shared_ptr<const Variable>, StateVariables>
  {
    assert(false && "Unimplemented");
  }
}
