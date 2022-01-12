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

  Eigen::VectorXd ConvolutionalLayer::feedForward()
  {
    Eigen::VectorXd output = Eigen::VectorXd::Zero(this->outputSize());

    // Well, we do convolutions
    for(size_t inputChannelIndex = 0; inputChannelIndex<m_inputChannelCount; ++inputChannelIndex)
      for(size_t outputChannelIndex = 0; outputChannelIndex<m_outputChannelCount; ++outputChannelIndex)
      {
        const auto outputWidth = m_inputWidth-m_kernelSize+1;
        const auto outputHeight = m_inputHeight-m_kernelSize+1;

        Eigen::Map<const Eigen::MatrixXd> inputChannel (input().data() + inputChannelIndex  * m_inputWidth * m_inputHeight, m_inputHeight, m_inputWidth);
        Eigen::Map<Eigen::MatrixXd>       outputChannel(output.data()  + outputChannelIndex * outputWidth  * outputHeight,  outputHeight,  outputWidth);

        const auto& kernel = m_kernels[inputChannelIndex * m_outputChannelCount + outputChannelIndex];

        for(size_t i=0; i<outputWidth; ++i)
          for(size_t j=0; j<outputHeight; ++j)
            outputChannel(j,i) += inputChannel.block(j,i,m_kernelSize,m_kernelSize).cwiseProduct(kernel).sum();
      }

    return output;
  }

  Eigen::VectorXd ConvolutionalLayer::backPropagate()
  {
    Eigen::VectorXd inputGradient = Eigen::VectorXd::Zero(this->inputSize());

    for(size_t inputChannelIndex = 0; inputChannelIndex<m_inputChannelCount; ++inputChannelIndex)
      for(size_t outputChannelIndex = 0; outputChannelIndex<m_outputChannelCount; ++outputChannelIndex)
      {
        const auto outputWidth = m_inputWidth-m_kernelSize+1;
        const auto outputHeight = m_inputHeight-m_kernelSize+1;

        Eigen::Map<const Eigen::MatrixXd> inputChannel         (input().data()          + inputChannelIndex  * m_inputWidth * m_inputHeight, m_inputHeight, m_inputWidth);
        Eigen::Map<Eigen::MatrixXd>       inputGradientChannel (inputGradient.data()    + inputChannelIndex  * m_inputWidth * m_inputHeight, m_inputHeight, m_inputWidth);
        Eigen::Map<const Eigen::MatrixXd> outputGradientChannel(outputGradient().data() + outputChannelIndex * outputWidth  * outputHeight,  outputHeight,  outputWidth);

        const auto& kernel   = m_kernels[inputChannelIndex * m_outputChannelCount + outputChannelIndex];
        auto& kernelGradient = m_kernelsGradient[inputChannelIndex * m_outputChannelCount + outputChannelIndex];

        for(size_t i=0; i<outputWidth; ++i)
          for(size_t j=0; j<outputHeight; ++j)
          {
            inputGradientChannel.block(j,i,m_kernelSize,m_kernelSize) += outputGradientChannel(j,i) * kernel;
            kernelGradient                                            += outputGradientChannel(j,i) * inputChannel.block(j,i,m_kernelSize,m_kernelSize);
          }
      }

    return inputGradient;
  }

  std::vector<std::span<double>> ConvolutionalLayer::params()
  {
    std::vector<std::span<double>> params;
    std::transform(m_kernels.begin(), m_kernels.end(), std::back_inserter(params), [](Eigen::MatrixXd& kernel) {
      return std::span<double>(kernel.data(), kernel.size());
    });
    return params;
  }

  std::vector<std::span<const double>> ConvolutionalLayer::params() const
  {
    std::vector<std::span<const double>> params;
    std::transform(m_kernels.begin(), m_kernels.end(), std::back_inserter(params), [](const Eigen::MatrixXd& kernel) {
      return std::span<const double>(kernel.data(), kernel.size());
    });
    return params;
  }

  std::vector<std::span<double>> ConvolutionalLayer::paramsGradient()
  {
    std::vector<std::span<double>> paramsGradient;
    std::transform(m_kernelsGradient.begin(), m_kernelsGradient.end(), std::back_inserter(paramsGradient), [](Eigen::MatrixXd& kernelGradient) {
      return std::span<double>(kernelGradient.data(), kernelGradient.size());
    });
    return paramsGradient;
  }

  std::vector<std::span<const double>> ConvolutionalLayer::paramsGradient() const
  {
    std::vector<std::span<const double>> paramsGradient;
    std::transform(m_kernelsGradient.begin(), m_kernelsGradient.end(), std::back_inserter(paramsGradient), [](const Eigen::MatrixXd& kernelGradient) {
      return std::span<const double>(kernelGradient.data(), kernelGradient.size());
    });
    return paramsGradient;
  }
}
